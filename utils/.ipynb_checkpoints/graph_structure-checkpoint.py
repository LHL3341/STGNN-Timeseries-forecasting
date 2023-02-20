import torch
import networkx as nx
import matplotlib.pyplot as plt

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from PyEMD import EMD
import math


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
    emd.emd(signal)
    #emd.emd(signal, np.arange(len(signal)))
    imfs, res = emd.get_imfs_and_residue()
    imfs_dict = {}
    for _ in range(imfs.shape[0]):
        imfs_dict[f'imf_{_}'] = imfs[_].tolist()
    res=res.tolist()
    return imfs_dict,res


def calculate_imf_features(n_zones, index_pair_for_one, zones_dict: dict, ticker_SMD_dict: dict,n_imf_use=5) -> np.ndarray:
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


def construct_res_graph(n_zones,res,adj_mat_method='correlation'):
    if adj_mat_method == 'fully_connected':
        # all 1 except the diagonal
        adj_mat = np.ones((n_zones, n_zones)) - np.eye(n_zones)  # np.eye()单位矩阵，对角线为1，其余0
    elif adj_mat_method == 'correlation':
        # based on correlation
        correlation_matrix = np.abs(np.corrcoef(np.array(res).T, rowvar=False))
        # np.where(expression,x,y)   expression成立，返回x；否则返回y
        correlation_matrix = np.where(correlation_matrix == 1., 0, correlation_matrix)
        correlation_matrix = np.where(correlation_matrix >= 0.9, 1, 0)  # 0.75
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


def construct_imfs_graph(x,zones,adj_mat):
    x = x.numpy()
    zones = range(zones)
    n_zones, zones_dict = len(zones), dict(zip(range(len(zones)), zones))
    ## calculate imf_features
    index_pair_for_one = np.argwhere(
        np.triu(adj_mat) == 1)  # get the index pair form upper triangle part of adj_mat 上三角有联系（=1）的索引
    ticker_SMD_dict = dict.fromkeys(zones)  # 创建并返回一个新的字典。两个参数：第一个是字典的键，第二个（可选）是传入键的值，默认为None。
    involde_index_idxs_np = np.unique(index_pair_for_one.flatten())  # 先降维到一维，然后去重按小到大排序
    # 根据关联矩阵来对某个特征下时间序列做IMF
    for index_idx in involde_index_idxs_np:
        ticker = zones_dict[index_idx]
        ticker_SMD_dict[ticker], _ = emd_imf_res(x[ticker].tolist())  # SMD_dict[0]=(8,720),SMD_dict[1-6]=(7,720)

    imf_features = calculate_imf_features(n_zones, index_pair_for_one, zones_dict, ticker_SMD_dict,
                                          n_imf_use=5)  # (7,7,5)

    # split the imd features into dict
    imf_matries_dict = {
        'imf_1_matix': imf_features[:, :, 0],  # (7,7)np数组,即可看做每个IMF分量的特征图
        'imf_2_matix': imf_features[:, :, 1],
        'imf_3_matix': imf_features[:, :, 2],
        'imf_4_matix': imf_features[:, :, 3],
        'imf_5_matix': imf_features[:, :, 4],
    }
    return imf_matries_dict


def matrix_to_edge_index(nodenum,adj_matrix):
    i_idx=[]
    j_idx=[]

    for i in range(nodenum):
        for j in range(nodenum):
            if adj_matrix[i][j]==1:
                i_idx.append(i)
                j_idx.append(j)
    i_idx=torch.tensor([i_idx])
    j_idx=torch.tensor([j_idx])
    edge_idx = torch.cat((i_idx, j_idx), dim=0)
    return edge_idx


def construct_node_graph(nodenum,type):
    if type =='full':
        i_idx = torch.arange(0, nodenum).T.unsqueeze(1).repeat(1, nodenum).flatten().to('cpu').unsqueeze(0)
        j_idx = torch.arange(0, nodenum).repeat(1, nodenum).to('cpu')
        edge_idx = torch.cat((i_idx,j_idx),dim=0)
    if type == 'cor':
        pass
    return edge_idx

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(b,2, edge_num)
    edge_idx= org_edge_index[0,:,np.argwhere(org_edge_index[0,0,:]!=-1)].squeeze(1)
    for i in range(1,batch_num):
        index_i= org_edge_index[i,:,np.argwhere(org_edge_index[i,0,:]!=-1)].squeeze(1) + i*node_num 
        edge_idx = np.concatenate([edge_idx,index_i],axis=1)
    return torch.tensor(edge_idx).long()

def get_batch_index(org_edge_index, batch_num):
    # org_edge_index:(2, node_num**2)
    edge_idx = org_edge_index
    node_num = math.sqrt(org_edge_index.shape[1])
    for i in range(1,batch_num):
        index_i= org_edge_index + i*node_num 
        edge_idx = torch.cat([edge_idx,index_i],dim=1)
    return edge_idx.long()