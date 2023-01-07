import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from PyEMD import EMD


def emd_imf_res(signal):
    """
    This function is to calculate EMD of the time series.
    :params: signal: list

    :return: res_dict: a dict consists of the different imf list value
    """
    if isinstance(signal, list):
        signal = np.array(signal)
    assert isinstance(signal, np.ndarray)
    emd = EMD()
    emd.emd(signal, np.arange(len(signal)))
    imfs, res = emd.get_imfs_and_residue()
    imfs_dict = {}
    for _ in range(imfs.shape[0]):
        imfs_dict[f'imf_{_}'] = imfs[_].tolist()
    res=res.tolist()
    return imfs_dict,res


def calculate_imf_features(n_zones, index_pair_for_one, zones_dict: dict, ticker_SMD_dict: dict,n_imf_use=3) -> np.ndarray:
    """
    compute the EMD.

    :return: imf_features
    """
    assert isinstance(n_imf_use, int)
    imf_features = np.zeros((n_zones, n_zones, n_imf_use))
    ticker_A, ticker_B = None, None
    for pair in index_pair_for_one:
        if ticker_A != zones_dict[pair[0]]:
            ticker_A = zones_dict[pair[0]]
            ticker_A_SMD = ticker_SMD_dict[ticker_A]
        if ticker_B != zones_dict[pair[1]]:
            ticker_B = zones_dict[pair[1]]
            ticker_B_SMD = ticker_SMD_dict[ticker_B]

        ef = [0] * n_imf_use
        for n_imf in list(range(1, n_imf_use + 1)):  # n_imf_to_exact = n_imf_use
            if f'imf_{n_imf}' in ticker_A_SMD and f'imf_{n_imf}' in ticker_B_SMD:
                # to get both imf for both 2 tickers
                ef[n_imf - 1] = (np.corrcoef(ticker_A_SMD[f'imf_{n_imf}'],
                                             ticker_B_SMD[f'imf_{n_imf}'])[0][1]
                )
            else:  # exit the loop when there is no further imf correctlation
                break
        imf_features[pair[0]][pair[1]], imf_features[pair[1]][pair[0]] = np.array(ef), np.array(ef)

    return imf_features


def construct_graph(n_zones,res,adj_mat_method):
    if adj_mat_method == 'fully_connected':
        # all 1 except the diagonal
        adj_mat = np.ones((n_zones, n_zones)) - np.eye(n_zones)  # np.eye()单位矩阵，对角线为1，其余0
    elif adj_mat_method == 'correlation':
        # based on correlation
        correlation_matrix = np.abs(np.corrcoef(res, rowvar=False))
        # np.where(expression,x,y)   expression成立，返回x；否则返回y
        correlation_matrix = np.where(correlation_matrix == 1., 0, correlation_matrix)
        correlation_matrix = np.where(correlation_matrix >= 0.8, 1, 0)  # 0.75
        adj_mat = correlation_matrix
    elif adj_mat_method == 'zero_mat':
        # zero matrix
        adj_mat = np.zeros((n_zones, n_zones))
    elif adj_mat_method == 'random':
        # random
        b = np.random.random_integers(0, 1, size=(n_zones, n_zones))
        adj_mat = b * b.T
    else:
        raise TypeError(f'Unsupported adj_matrix method: {adj_mat_method}!')

    return adj_mat