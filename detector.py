import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim,nn
import torch.nn.functional as F
from dataset.data_loader import get_dataloader
from model.model import Model
from utils.graph_structure import show_graph
import time
def loss_fn(y_pred,y_recon, y_true,x):
    p_loss = F.mse_loss(y_pred.squeeze(-1), y_true, reduction='mean')
    r_loss = F.mse_loss(y_recon, x, reduction='mean')
    return p_loss,r_loss

class Detector(object):
    DEFAULTS = {}           

    def __init__(self, config):

        self.__dict__.update(Detector.DEFAULTS, **config)
        self.trainloader = get_dataloader(self.dataset, 'train',config)
        self.testloader = get_dataloader(self.dataset, 'test',config)
        self.valiloader = get_dataloader(self.dataset, 'vali',config)
        self.model = Model(config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = loss_fn

        if self.device == 'cuda':
            self.model.cuda()

        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()
            
    def train(self):
        self.model.train()
        now = time.time()
        
        train_r_loss_list = []
        train_p_loss_list = []
        cmp_loss_list = []

        acu_loss = 0
        min_loss = 1e+8
        min_f1 = 0
        min_pre = 0
        best_prec = 0

        dataloader = self.trainloader
        epoch = self.num_epochs
        device = self.device
        optimizer = self.optimizer
        model = self.model
        gamma = self.gamma

        for i_epoch in range(epoch):

            acu_loss = 0

            for x, y,labels,edge_idx in dataloader:
                _start = time.time()

                x, y,labels = [item.float().to(device) for item in [x, y,labels]]

                optimizer.zero_grad()
                r_out,p_out,attn_w = model(x,edge_idx)
                p_out = p_out.float().to(device)
                r_out = r_out.float().to(device)
                p_loss,r_loss = loss_fn(p_out,r_out, y,x)
                loss = gamma*p_loss+(1-gamma)*r_loss
                loss.backward()
                optimizer.step()

                train_r_loss_list.append(r_loss.item())
                train_p_loss_list.append(p_loss.item())
                acu_loss += loss.item()

            # each epoch
            print('epoch ({} / {}) (r_Loss:{:.8f},p_Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                            i_epoch, epoch, 
                            r_loss,p_loss, acu_loss), flush=True
                )
            show_graph(edge_idx,i_epoch,self.dataset,self.save_path)
            if (i_epoch+1)%10 ==0:
                torch.save(model.state_dict(), self.save_path+f'/{self.dataset}/epoch{i_epoch}.pkl')

    def vali():
        pass
    def test():
        pass