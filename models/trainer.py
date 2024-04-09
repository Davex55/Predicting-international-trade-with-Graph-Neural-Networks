# @Time     : Jan. 13, 2019 20:16
# @Author   : Veritas YIN
# @FileName : trainer.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin
from datetime import datetime

# Arreglo para que tensorflow funcione en v1, ya que parece que todo se ha hecho en esa versión. +
import tensorflow
from tensorflow.compat import v1 as tf

import numpy as np
import time

## Añadido!
import pandas as pd


def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt

    # Arreglo para quitar el modo "eager" +
    tf.disable_eager_execution()

    # Placeholder for model training, x is  empthy!
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Define model loss
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)


    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())

        # step_idx determines which prediction is to be evaluated, min_val store the
        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = np.array([4e20, 1e20, 1e20]) #np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge': # Por defecto
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = np.array([4e20, 1e20, 1e20] * len(step_idx)) #np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        # This is where the training happens
        for i in range(epoch):
            t1 = time.time()
            # x_batch are simply train chunks of size ["batch_size", 21, 228, 1].
            # In addition, the loop is going to run len(inputs.get_data('train'))/batch_size, which for the standard case is 182-183 times.
            # n_his, refers to the number of historical records to be used for the prediction, by default 13 of the possible 21 possible records are used.
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                # Train_op = Train optimizers

                # Enter data to make predictions.
                # Calculates the train_loss with the X (originally 8) rows that it does not use.
                # As the "train_op" is entered, this is where the backpropagation is done, because for that you need an optimizer!!!
                # Reminder, that the last of the x's is used as the y, i.e. if from [0:n_his + 1] go thirteen,
                    # 12 will be used to make a prediction and this will be compared to the remaining thirteen to calculate the loss.
                #TODO comprobar si se puede mejorar el proceso de entrenamiento (se entrena con npred = 1 y se desperdicia el resto)
                summary, w = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1})
                writer.add_summary(summary, i * epoch_step + j)

                # Check every 10 iterations!
                if j % 10 == 0: # if j % 10 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')

            t2 = time.time()

            Info = "Epoch: {epoch}, Training Time: {time}"
            print(Info.format(epoch = i, time = time.strftime("%H:%M:%S", time.gmtime(t2 - t1))))

            if(inputs.get_len('val') > 0):

                t1 = time.time()

                min_val = model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_val)
                for ix in tmp_idx:
                    va = min_val[ix - 2:ix + 1]
                    print(f'Time Step {ix + 1}: '
                        f'MAPE {va[0]:7.3%}; '
                        f'MAE  {va[1]:4.3f}; '
                        f'RMSE {va[2]:6.3f}.')

                t2 = time.time()

                Info = "Epoch: {epoch}, Inference Time: {time}"
                print(Info.format(epoch = i, time = time.strftime("%H:%M:%S", time.gmtime(t2 - t1))))

                summary = tf.Summary(value=[tf.Summary.Value(tag='Validation Loss', simple_value=min_val[2])])

            # add the summary to the writer
            writer.add_summary(summary, global_step=i)

            if (i + 1) % args.save == 0:
                model_save(sess, global_steps, 'STGCN')

        writer.flush()
        writer.close()
    print('Training model finished!')
