import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

from graph_structure import construct_graph

class TimeDataset(Dataset):
    DEFAULTS = {}
    def __init__(self,config = None):
        self.__dict__.update(TimeDataset.DEFAULTS, **config)
        data = pd.read_csv(f'./data/{self.dataset}/{self.mode}.csv', sep=',', index_col=0,)
        data = np.array(data)
        x = data[:, :-1]
        if self.mode == 'train':
            label = np.zeros(data.shape[0])
        else:
            label = data[:,-1]
        # to tensor
        x= torch.tensor(x).double()
        label = torch.tensor(label).double()

        self.x, self.y, self.labels = self.process(x, label)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = construct_graph(y.shape[-1],self.graph_type)

        label = self.labels[idx].double()

        return feature, y, label, edge_index

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = self.win_size,1
        is_train = self.mode == 'train'

        total_time_len,node_num = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        for i in rang:

            ft = data[ i-slide_win:i,:]
            tar = data[ i,:]

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])


        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, labels

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
