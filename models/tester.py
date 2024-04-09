# @Time     : Jan. 10, 2019 17:52
# @Author   : Veritas YIN
# @FileName : tester.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion


from utils.math_utils import evaluation, MAE, descale
from data_loader.data_utils import gen_batch
from tensorflow.compat import v1 as tf
from os.path import join as pjoin

import pandas as pd
import numpy as np
import time



def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, number of historical records used as input for the model.
    :param n_pred: int, the number of predictions.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, [n_pred, len(seq), n_route, C_0].
            len_ : int, len(seq).
    '''
    pred_list = []
    # Note: when the batch_size is greater than the length of the seq array, the gen_batch function uses len(seq) as batch size
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        step_list = []

        # n_pred predictions are made
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})

            if isinstance(pred, list):
                pred = np.array(pred[0])

            # The test_seq data is updated for the following prediction step
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred

            # The results of the prediction are added to step_list
            step_list.append(pred)

        # Predictions of this batch execution are added to pred_list
        pred_list.append(step_list)

    # pred_array -> [n_pred, len(seq), n_route, C_0]
    pred_array = np.concatenate(pred_list, axis=1)

    return pred_array, pred_array.shape[1]



def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_val):
    '''
    Model inference function.
    :param sess: tf.Session().
    :param pred: placeholder.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, number of historical records used as input for the model.
    :param n_pred: int, the number of predictions.
    :param step_idx: int or list, index for the evaluation of predictions.
    :param min_val: np.ndarray, metric values on validation set.
    '''
    # Validation dataset, dataset statistics and normalization function applied to dataset
    x_val, normalization, stats = inputs.get_data('val'), inputs.get_normalization(), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')

    # Predictions of the validation dataset
    y_val, len_val = multi_pred(sess, pred, x_val, batch_size, n_his, n_pred)
    # Evaluate the predictions of y_val with the ground truth data stored in x_val.
    # Note: with step_idx + n_his we get the index of the prediction store in x_val
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val[step_idx], normalization, stats)
    # chks: indicator that reflects the relationship of values between evl_val and min_val (array of booleans)
    chks = evl_val < min_val
    # Update the metric on validation set if model's performance got improved
    min_val[chks] = evl_val[chks]

    return min_val



def model_test(inputs, batch_size, n_his, n_pred, inf_mode, load_path='./output/models/'):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, number of historical records used as input for the model.
    :param n_pred: int, the number of predictions.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    # Timer
    t1 = time.time()

    # Load saved model from a checkpoint
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        # Get the 'y_pred' tensor from the tf graph collection
        pred = test_graph.get_collection('y_pred')
        print(type(pred))

        # Selection of the information mode. The step_idx variable determines which prediction is to be evaluated
        if inf_mode == 'sep':
            # Note: for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # Note: for inference mode 'merge', the type of step index is np.ndarray (may have more than 1 prediction to evaluate).
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        # Test dataset, dataset statistics and normalization function applied to dataset
        x_test, normalization, stats = inputs.get_data('test'), inputs.get_normalization(), inputs.get_stats()

        # Predictions of the test dataset
        y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred)
        # Evaluate the predictions of y_test with the ground truth data stored in x_test.
        # Note: if inf_mode == 'sep' x_test -> [len(len_test), n_route, C_0], else x_test -> [len(len_test), len(step_idx), n_route, C_0]
        evl = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test[step_idx], normalization, stats)

        # Note: [n_pred, len(seq), n_route, C_0] -> [len(seq), n_pred, n_route, C_0]
        y_test = np.swapaxes(y_test, 0, 1)
        # Save model results in externals csv files
        save_results(len_test, y_test[0:len_test, :, :, 0], normalization, stats)
        save_results(len_test, x_test[0:len_test, n_his:, :, 0], normalization, stats)

        # Show evaluation results
        for ix in tmp_idx:
            te = evl[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: '
                        f'MAPE {te[0]:7.3%}; '
                        f'MAE  {te[1]:4.3f}; '
                        f'RMSE {te[2]:6.3f}.')

        # Timer
        t2 = time.time()
        Info = "Model Test Time: {time}"
        print(Info.format(time = time.strftime("%H:%M:%S", time.gmtime(t2 - t1))))

    print('Testing model finished!')


#TODO testear esto
def save_results(len_seq, seq, normalization, stats, path=''):
    '''
    Store a descaled np.array in a .csv file.
    :param len_seq: int , length of seq np.array.
    :param seq: int, the length of historical records for training.
    :param normalization: string, normalization function.
    :param stats: dict, parameters for normalize and denormalize the dataset (mean & std/irq).
    :param path: string, .
    '''
    for i in range(len_seq):
        # seq[i, :, :] = [n_pred, n_route]
        x_guardarZ = pd.DataFrame(descale(seq[i, :, :], stats['mean'], stats['std'], normalization))
        print(x_guardarZ.shape)
        x_guardarZ.to_csv(f'X_guardado_{i}Z.csv', header = None, index = False)