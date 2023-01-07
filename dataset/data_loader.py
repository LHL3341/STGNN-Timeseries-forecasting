import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


from utils.graph_structure import emd_imf_res, construct_node_graph, construct_res_graph, construct_imfs_graph, matrix_to_edge_index

class TimeDataset(Dataset):
    DEFAULTS = {}
    def __init__(self,config = None):
        self.__dict__.update(TimeDataset.DEFAULTS, **config)

        df = pd.read_csv(f'./data/{self.dataset}/{self.mode}.csv', sep=',', index_col=0,)
        slide_win = self.win_size
        is_train = self.mode == 'train'
        predict_len=self.predict_len
        total_time_len = df.shape[0]
        label = df.iloc[:, -1]   # 最后一列得到label
        df=df.iloc[:, :-1]  # df去掉最后的label
        zones = df.columns.to_list()  # 得到df的column，即由feature+label组成的列表
        self.zones = zones
        self.n_zones = len(self.zones)
        # print(self.zones)

        # 数据归一化
        self.scaler = StandardScaler()
        data = np.array(df)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        """
        # SR变换，数据增强
        if self.sr:
            # less than period
            amp_window_size = 24
            # (maybe) as same as period
            series_window_size = 24
            # a number enough larger than period
            score_window_size = 100  # 论文1440
            # %%
            spec = anom.Silency(amp_window_size, series_window_size, score_window_size)
            # %%
            data = spec.generate_anomaly_score(data)
        """

        if self.mode == 'train':
            label = np.zeros(data.shape[0])
        else:
            label = np.array(label)
        label = torch.tensor(label).double()
        data = torch.tensor(data).double()

        print('###################preprocessing################')
        x_list=[]
        y_list=[]
        labels_list=[]
        res_graphs_list=[]
        imf_graphs_list=[]
        rang = range(slide_win, total_time_len-predict_len)
        for i in rang:
            feature = df[i-slide_win:i]  #df
            ft = data[ i-slide_win:i,:]
            tar = data[i:i+predict_len,:]
            res_data=[]
            imfs_data=[]
            for zone in zones:
                signal=feature[zone]
                imfs_dic,res=emd_imf_res(np.array(signal))
                res_data.append(res)
                imfs_data.append(imfs_dic)

            res_graph=construct_res_graph(self.n_zones,res_data,adj_mat_method='correlation')
            imf_graph=construct_imfs_graph(feature,zones,res_graph)

            res_graphs_list.append(res_graph)
            imf_graphs_list.append(imf_graph)
            x_list.append(ft)
            y_list.append(tar)

            labels_list.append(label[i])

        x = torch.stack(x_list).contiguous()
        y = torch.stack(y_list).contiguous()

        labels = torch.Tensor(labels_list).contiguous()

        self.x = x
        self.y = y
        self.labels = labels
        self.res_graphs=res_graphs_list
        self.imf_graphs=imf_graphs_list
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        n_zones = self.n_zones
        feature = self.x[idx].double()  # tensor
        y = self.y[idx].double()
        label = self.labels[idx].double()

        res_matrix=self.res_graphs[idx]
        res_edge_index=matrix_to_edge_index(n_zones, res_matrix)
        node_edge_index = construct_node_graph(n_zones,self.graph_type)
        res_edge_index = np.pad(res_edge_index,((0,0),(0,n_zones**2-res_edge_index.shape[1])),'constant',constant_values=(-1,-1))
        return feature, y, label, node_edge_index, res_edge_index


def get_dataloader(dataset, mode,config):
    batch_size = config['batch_size']
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    if mode == 'test':
        batch_size = 1
    data = TimeDataset(config)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,num_workers=0)
