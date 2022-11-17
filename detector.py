import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim,nn
import torch.nn.functional as F
from dataset.data_loader import get_dataloader
from model.model import Model
from utils.graph_structure import show_graph
import time
from utils.time import timeSincePlus
from utils.test import get_best_performance_data,get_full_err_scores

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
            self.model.to(self.device)
            self.model.cuda()
            
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
            r_loss_l = []
            p_loss_l = []
            for x, y,labels,edge_idx in dataloader:
                _start = time.time()

                x, y = [item.float().to(device) for item in [x, y]]

                optimizer.zero_grad()
                r_out,p_out,attn_w = model(x,edge_idx)
                p_out = p_out.float().to(device)
                r_out = r_out.float().to(device)
                p_loss,r_loss = loss_fn(p_out,r_out, y,x)
                loss = gamma*p_loss+(1-gamma)*r_loss
                loss.backward()
                optimizer.step()

                r_loss_l.append(r_loss.item())
                p_loss_l.append(p_loss.item())
                acu_loss += loss.item()
            train_p_loss_list.append(np.mean(p_loss_l))
            train_r_loss_list.append(np.mean(r_loss_l))
            # each epoch
            print('epoch ({} / {}) (r_Loss:{:.8f},p_Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                            i_epoch, epoch, 
                            r_loss,p_loss, acu_loss), flush=True
                )
            #show_graph(edge_idx,i_epoch,self.dataset,self.save_path)
            
            plt.plot(range(i_epoch+1),train_p_loss_list)
            plt.plot(range(i_epoch+1),train_r_loss_list)
            plt.show()
            if (i_epoch+1)%10 ==0:
                torch.save(model.state_dict(), self.save_path+f'/{self.dataset}/epoch{i_epoch}.pkl')

    def vali():
        pass
    def test(self):
        device = self.device
        dataloader = self.testloader
        self.model.load_state_dict(torch.load(self.save_path+f'/{self.dataset}/epoch{self.pretrained_epochs}.pkl'))
        model = self.model
        if self.device == 'cuda':
            self.model.to(device)
            self.model.cuda()
        now = time.time()

        test_predicted_list = []
        test_recon_l_list = []

        t_test_predicted_list = []
        t_test_ground_list = []
        t_test_labels_list = []

        test_len = len(dataloader)

        model.eval()

        i = 0
        acu_loss = 0
        for i,(x, y, labels, edge_index) in enumerate(dataloader):
            for j in range(x.shape[0]+x.shape[2]):
                    test_recon_l_list.append([])
            x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
            
            with torch.no_grad():
                recon,predicted,attn_w = model(x, edge_index)
                predicted = predicted.float().to(device)
                recon = recon.float().to(device)
                p_loss,r_loss = loss_fn(predicted,recon, y,x)
                labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])
                for j in range(x.shape[0]):
                    idx = i*dataloader.batch_size+j
                    for k in range(x.shape[2]):
                        l = torch.nn.MSELoss()(x[j,:,k],recon[j,:,k])
                        test_recon_l_list[idx+k].append(l.item())

                if len(t_test_predicted_list) <= 0:
                    t_test_predicted_list = predicted
                    t_test_ground_list = y
                    t_test_labels_list = labels
                else:
                    t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                    t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                    t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

            if i % 10000 == 1 and i > 1:
                print(timeSincePlus(now, i / test_len))

        test_predicted_list = t_test_predicted_list.tolist()
        test_recon_l_list = test_recon_l_list[1:len(test_predicted_list)+1]
        test_ground_list = t_test_ground_list.tolist()        
        test_labels_list = t_test_labels_list.tolist()
        self.get_score([test_predicted_list,test_ground_list,test_labels_list],test_recon_l_list)


    def get_score(self, test_result,test_recon):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores = get_full_err_scores(test_result)
        t_recon_ = [np.mean(l) for l in test_recon]
        t_recon = np.array(t_recon_).reshape(1,-1)
        #test_scores = test_scores + t_recon
        
        info = get_best_performance_data(test_scores, test_labels, topk=1) 


        print('=========================** Result **============================\n')

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')
