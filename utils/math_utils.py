# @Time     : Jan. 10, 2019 15:15
# @Author   : Veritas YIN
# @FileName : math_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import numpy as np
import pandas as pd
import math


########################### Normalization Functions ###########################
###############################################################################


def z_score(x, mean, std):
    '''
    Z-score normalization function.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_score_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return (x * std) + mean


def robust_scale(x, mean, iqr):
    '''
    Robust normalization function.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized array.
    '''

    return (x - mean) / iqr


def inverse_robust_scale(x, mean, iqr):
    '''
    The inverse of function robust_scale().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust scale inverse array.
    '''

    return (x * iqr) + mean


def log_scale(x):
    '''
    Log scaling normalization function.
    :param x: np.ndarray, input array to be normalized.
    :return: np.ndarray, Log scale normalized array.
    '''

    return np.log1p(x)


def inverse_log_scale(x):
    '''
    The inverse of function log_scale().
    :param x: np.ndarray, input array to be normalized.
    :return: np.ndarray, Log scale inversed array.
    '''

    return np.expm1(x)


##############################################################################
################## Normalization Additional Functionalities ##################
##############################################################################


def scale(x, stats, normalize_type="z_score"):
    '''
    Normalization of the data
    :param x: np.ndarray, input array to be normalized.
    :param stats: dict, parameters for normalize and denormalize the dataset (mean & std/iqr).
    :param normalize_type: string, normalization function to be used.
    :return: np.ndarray, normalized np.ndarray.
    '''
    if normalize_type == "z_score":
        return z_score(x, stats['mean'], stats['std'])
    elif normalize_type == "robust":
        return robust_scale(x, stats['mean'], stats['iqr'])
    elif normalize_type == "log_scale":
        return log_scale(x)
    elif normalize_type == "none":
        return x
    else:
        raise ValueError("No accepted scalation type!!")


def descale(x, stats, normalize_type="z_score"):
    '''
    Denormalization of the data
    :param x: np.ndarray, input array to be denormalized.
    :param stats: dict, parameters for normalize and denormalize the dataset (mean & std/iqr).
    :param normalize_type: string, normalization function to be used.
    :return: np.ndarray, normalized inversed np.ndarray.
    '''
    if normalize_type == "z_score":
        return z_score_inverse(x, stats['mean'], stats['std'])
    elif normalize_type == "robust":
        return inverse_robust_scale(x, stats['mean'], stats['iqr'])
    elif normalize_type == "log_scale":
        return inverse_log_scale(x)
    elif normalize_type == "none":
        return x
    else:
        raise ValueError("No accepted scalation type!!.")


def gen_stats(data, normalize_type):
    '''
    Dataset stats generation (mean, std and iqr)
    :param data: np.ndarray, input array to be normalized.
    :param normalize_type: string, normalization function to be used.
    :return: dict, dictionary with different stats depending on the normalization function.
    '''
    if normalize_type == "z_score" or normalize_type == "none":
        return {'mean': np.mean(data), 'std': np.std(data)}
    elif normalize_type == "robust":
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        return {'mean': np.mean(data),'iqr': iqr}
    elif normalize_type == "log_scale":
        return {}
    else:
        raise ValueError("No accepted scalation type!!.")


##############################################################################
############### Evaluation Between Ground Truth And Prediction ###############
##############################################################################


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs((v_ - v) / (v + 1e-5)))


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, normalization, stats):
    '''
    Interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param normalization: string, normalization function.
    :param stats: dict, parameters for normalize and denormalize the dataset (mean & std/iqr).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)

    if dim == 3:
        # single_step case

        # It is better to keep the y and y_ variables normalized because of their high values.
        # v = descale(y, stats, normalization)
        # v_ = descale(y_, stats, normalization)

        #TODO devolver los resultados de funciones de evaluacion con los valores normalizados y desnormalizados
        return np.array([MAPE(y, y_), MAE(y, y_), RMSE(y, y_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], normalization, stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)