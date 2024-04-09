# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from utils.math_utils import scale, gen_stats

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, normalization, stats):
        self._data = data
        self._normalization = normalization
        #TODO cambiar los nombres mean y std
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self._data[type]

    def get_len(self, type):
        return len(self._data[type])

    def get_normalization(self):
        return self._normalization

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target data sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of the data in the data_seq.
    :param n_frame: int, the number of frame within a standard sequence unit,
        which contains n_his and n_pred (number of historical data and number of predictions per inference).
    :param n_route: int, the number of routes in the graph.
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    # An empty numpy array is created to store the data
    tmp_seq = np.zeros((len_seq, n_frame, n_route, C_0))
    # One element of the time series is store in tmp_seq for each len_seq
    for i in range(len_seq):
        sta = offset + i
        end = sta + n_frame
        print(sta, end)
        # The information is stored in the desired format.
        tmp_seq[i, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


#TODO testear esto
def gen_interpolation_points(data_seq, partern_slot=3):
    '''
    Generate interpolation points for the dataset.
    :param data_seq: np.ndarray, source data / time-series.
    :param partern_slot: int, number of new points created between two original data points.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    concat_list = [data_seq]

    # Creation of an empty dataframe with the same indexes as data_seq
    nan_df = pd.DataFrame(index = data_seq.index)
    for i in range(partern_slot):
        concat_list.append(nan_df)

    # The df's in the concat_list are concatenated so that the resulting df contains enough empty rows to use interpolation.
    # Note: sort_index is used to sort the df and reset the index numbers. This way the empty rows are placed in the middle of the original ones.
    data_seq = (pd.concat(concat_list).sort_index(kind='stable', ignore_index = True))
    # Interpolation is used to fill empty rows in df
    data_seq.interpolate(method ='linear', limit_direction ='forward', inplace = True)

    return data_seq


def data_gen(file_path, data_config, n_route, n_frame=21, interpolation = False):
    '''
    Source file load and dataset generation for training, validation and test.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_frame: int, the number of frame within a standard sequence unit,
        which contains n_his and n_pred (number of historical data and number of predictions per inference).
    :param n_route: int, the number of routes in the graph.
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    # The length of each data sequence (training, validation and test) and normalization function
    n_train, n_val, n_test, normalization = data_config

    # Load the data into a numpy array
    try:
        data_seq = pd.read_csv(file_path, header=0)
        # data_seq = pd.read_csv(file_path, header=None).values

    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # Generation of new data points by interpolation
    if (interpolation):
        data_seq = gen_interpolation_points(data_seq, 3)

    # Generation of each sequence
    seq_train = seq_gen(n_train, data_seq.values, 0, n_frame, n_route)
    seq_val = seq_gen(n_val, data_seq.values, n_train, n_frame, n_route)
    seq_test = seq_gen(n_test, data_seq.values, n_train + n_val, n_frame, n_route)

    # Temporal
    seq_val = seq_test

    # stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    stats = gen_stats(seq_train, normalization)

    # Normalization of each np.array: x_train, x_val, x_test
    x_train = scale(seq_train, stats['mean'], stats['std'], normalization)
    x_val = scale(seq_val, stats['mean'], stats['std'], normalization)
    x_test = scale(seq_test, stats['mean'], stats['std'], normalization)

    # Creation of the dataset object
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, normalization, stats)

    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    # Creation of unordered list of indexes
    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        # Get the start and end of batch for current iteration
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            # Obtaining the unordered batch indices for current iteration
            slide = idx[start_idx:end_idx]
        else:
            # Obtaining the batch indices for current iteration
            slide = slice(start_idx, end_idx)

        yield inputs[slide]