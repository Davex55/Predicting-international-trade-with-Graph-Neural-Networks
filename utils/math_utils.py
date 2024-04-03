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
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return ((x - mean) / std)*1


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return (x/1) * std + mean


def robust_scale(data, mean, irq):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized array.
    '''
    # Aplicar la fórmula de normalización robusta
    scaled_data = np.zeros_like(data)
    for c1 in range(len(data[:, 0, 0, 0])):
        for c3 in range(len(data[0, 0, :, 0])):
            for c4 in range(len(data[0, 0, 0, :])):
                scaled_data[c1, :, c3, c4] = (data[c1, :, c3, c4] - mean[c1, 0, c3, c4]) / irq[c1, 0, c3, c4]

    return scaled_data


def inverse_robust_scale(scaled_data, mean, irq):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized inversed array.
    '''
    # Realizar la inversión de la fórmula de normalización robusta
    data = np.zeros_like(scaled_data)
    for c1 in range(len(scaled_data[:, 0, 0, 0])):
        for c3 in range(len(scaled_data[0, 0, :, 0])):
            for c4 in range(len(scaled_data[0, 0, 0, :])):
                data[c1, :, c3, c4] = (scaled_data[c1, :, c3, c4] * irq[c1, 0, c3, c4]) + mean[c1, 0, c3, c4]

    return data


def robust_scale_original(x, mean, irq):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized array.
    '''
    scaled_data =  (x - mean) / irq

    return scaled_data


def inverse_robust_scale_original(x, mean, iqr):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized inversed array.
    '''
    inv_scaled_data = (x * iqr) + mean

    return inv_scaled_data


def log_scale(x, mean, std):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized array.
    '''
    normalized_data = np.log1p(x)

    return normalized_data


def inverse_log_scale(x, mean, std):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized inversed array.
    '''
    inv_normalized_data = np.expm1(x)

    return inv_normalized_data



##############################################################################
################## Normalization Additional Functionalities ##################
##############################################################################



def scale(x, p1, p2, normalize_type="robust_original"):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param p1: .
    :param p2: .
    :param scalation_type: .
    :return: np.ndarray, normalized np.ndarray.
    '''
    if normalize_type == "z_score":
        return z_score(x, p1, p2)
    elif normalize_type == "robust":
        return robust_scale(x, p1, p2)
    elif normalize_type == "robust_original":
        return robust_scale_original(x, p1, p2)
    elif normalize_type == "log_scale":
        return log_scale(x, p1, p2)
    elif normalize_type == "none":
        return x
    else:
        raise ValueError("No accepted scalation type!!")


def descale(x, p1, p2, normalize_type="robust_original"):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param p1: .
    :param p2: .
    :param scalation_type: .
    :return: np.ndarray, normalized inversed np.ndarray.
    '''
    if normalize_type == "z_score":
        return z_inverse(x, p1, p2)
    elif normalize_type == "robust":
        return inverse_robust_scale(x, p1, p2)
    elif normalize_type == "robust_original":
        return inverse_robust_scale_original(x, p1, p2)
    elif normalize_type == "log_scale":
        return inverse_log_scale(x, p1, p2)
    elif normalize_type == "none":
        return x
    else:
        raise ValueError("No accepted scalation type!!.")


def gen_stats(data, normalize_type):
    '''
    :param data: np.ndarray, input array to be normalized.
    :param scalation_type: .
    :return: np.ndarray, normalized np.ndarray.
    '''
    if normalize_type == "z_score" or normalize_type == "none":
        return {'mean': np.mean(data), 'std': np.std(data)}
    elif normalize_type == "robust":
        mean, std = get_column_data(data)
        return {'mean': mean, 'std': std}
    elif normalize_type == "robust_original":
        return {'mean': np.mean(data), 'std': np.percentile(data, 75) - np.percentile(data, 25)}
    elif normalize_type == "log_scale":
        return {'mean': np.mean(np.log1p(data)), 'std': np.std(np.log1p(data))}
    else:
        raise ValueError("No accepted scalation type!!.")

#TODO arreglar esto
def get_column_data(data):
    '''
    :param data: .
    :return: .
    :return: .
    '''
    aux_mean = np.zeros( (len(data[:, 0, 0, 0]), 1, len(data[0, 0, :, 0]), len(data[0, 0, 0, :])) )
    aux_iqr = np.zeros( (len(data[:, 0, 0, 0]), 1, len(data[0, 0, :, 0]), len(data[0, 0, 0, :])) )
    for c1 in range(len(data[:, 0, 0, 0])):
        for c3 in range(len(data[0, 0, :, 0])):
            for c4 in range(len(data[0, 0, 0, :])):
                aux_mean[c1, 0, c3, c4] = data[c1, :, c3, c4].mean()
                aux_iqr[c1, 0, c3, c4] = np.percentile(data[c1, :, c3, c4], 75) - np.percentile(data[c1, :, c3, c4], 25)
                if aux_iqr[c1, 0, c3, c4] == 0:
                        aux_iqr[c1, 0, c3, c4] = 1

    return aux_mean, aux_iqr



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
    # Crear un índice booleano para los valores infinitos
    v_aux = pd.Series(v_[0, :, 0])
    vaux = pd.Series(v[0, :, 0])
    data = (abs((v_aux - vaux)/(vaux))).replace(np.inf, 0)

    return np.mean(data)


def MAPE_antiguo(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v) / (v + 1e-5))


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
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param normalization: string, normalization function.
    :param stats: dict, parameters for normalize and denormalize the dataset (mean & std/irq).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)

    if dim == 3:
        # single_step case

        # It is better to keep the y and y_ variables normalized because of their high values.
        # v = descale(y, stats['mean'], stats['std'], normalization)
        # v_ = descale(y_, stats['mean'], stats['std'], normalization)

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