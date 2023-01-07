import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import sranodec as anom

def construct_graph(nodenum,type):
    if type =='full':
        i_idx = torch.arange(0, nodenum).T.unsqueeze(1).repeat(1, nodenum).flatten().to('cpu').unsqueeze(0)
        j_idx = torch.arange(0, nodenum).repeat(1, nodenum).to('cpu')
        edge_idx = torch.cat((i_idx,j_idx),dim=0)
    if type == 'cor':
        pass
    return edge_idx

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


def matrix_to_edge(n):
    a=[]
    b=[]
    for i in range(n):
        a.append(i)
        b.append(i)
    a=[a]
    b=[b]
    a=torch.tensor(a)
    b=torch.tensor(b)
    return a,b

def get_data():
    df = pd.read_csv(f'./data/msl/train.csv', sep=',', index_col=0, )
    # print(0)
    zones = df.columns.to_list()  # 得到df的column，即由feature+label组成的列表
    zones.pop()  # 去掉label
    # print(self.zones)
    slide_win, slide_stride = 50, 1
    is_train = True
    data = df.iloc[:, :-1]
    total_time_len = df.shape[0]
    data=np.array(data)
    amp_window_size = 24
    # (maybe) as same as period
    series_window_size = 24
    # a number enough larger than period
    score_window_size = 100  # 论文1440
    # %%
    spec = anom.Silency(amp_window_size, series_window_size, score_window_size)
    # %%
    data = spec.generate_anomaly_score(data)
    print(data)
    print(0)
    rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
    return data

    """
    data = np.array(df)
    x = data[:, :-1]
    if self.mode == 'train':
        label = np.zeros(data.shape[0])
    else:
        label = data[:, -1]
    # to tensor
    x = torch.tensor(x).double()
    label = torch.tensor(label).double()
    """

if __name__ == "__main__":
    """
    edge=construct_graph(5,"full")
    batch_edge=get_batch_edge_index(edge,2,5)

    print(edge)
    print(batch_edge)
    """
    #a,b=matrix_to_edge(7)
    data=get_data()



