import numpy as np
import pandas as pd
from pandas import DataFrame as df
import torch
import matplotlib.pyplot as plt
from torch import optim,nn
import torch.nn.functional as F
from dataset.data_loader import get_dataloader
from model.model import Model
import time
from utils.time import timeSincePlus
from utils.test import get_best_performance_data,get_full_err_scores
from utils.log import excel_logs,plot_loss,plot_pred
import os
import warnings
#warnings.filterwarnings("ignore")

def loss_fn(y_pred,y_recon, y_true,x):
    feature_num = y_true.shape[-1]
    y_true,y_pred=y_true.reshape(-1,feature_num),y_pred.reshape(-1,feature_num)
    mse_loss = F.mse_loss(y_pred, y_true, reduction='mean')
    mae_loss = F.l1_loss(y_true,y_pred)
    #r_loss = F.mse_loss(y_recon, x, reduction='mean')
    return mse_loss,mae_loss

class Solver(object):
    DEFAULTS = {}           

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)
        self.trainloader = get_dataloader('train',0,config)
        if self.mode=='train':
            self.valiloader = get_dataloader('vali',self.trainloader.dataset.scaler,config)
        else:
            print(self.trainloader.dataset.scaler)
            self.testloader = get_dataloader('test',self.trainloader.dataset.scaler,config)
        self.model = Model(config)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.model.parameters())))
        #print(self.model._modules)
        print('tcn',sum(x.numel() for x in self.model._modules['tcn'].parameters()))
        print('gat',sum(x.numel() for x in self.model._modules['gat'].parameters()))
        print('mlp',sum(x.numel() for x in self.model._modules['f'].parameters()))
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,weight_decay = self.wd)
        print('wd',self.wd)
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[30,50,80], gamma=0.2)#20,30,40,45
        self.criterion = loss_fn
            
    def train(self):
        print('###################training################')
        
        now = time.time()
        
        train_mse_list = []
        train_mae_list = []
        val_mse_list = []
        val_mae_list= []

        acu_loss = 0
        min_loss = 1e+8
        min_f1 = 0
        min_pre = 0
        best_prec = 0

        dataloader = self.trainloader
        epoch = self.num_epochs
        device = self.device
        optimizer = self.optimizer
        scheduler = self.scheduler
        model = self.model
        gamma = self.gamma
        start = time.time()
        for i_epoch in range(epoch):
            self.model.train()
            iter = 0
            acu_loss = 0
            mse = []
            mae = []
            for x, y,node_edge_idx,res_edge_idx in dataloader:
                #print(x.shape)
                b,t,n,c=x.shape
                node_edge_idx,res_edge_idx = node_edge_idx.squeeze(1),res_edge_idx.squeeze(1)
                x, y = [item.float().to(device) for item in [x, y]]

                optimizer.zero_grad()
                r_out,p_out,attn_w1,attn_w2 = model(x,node_edge_idx,res_edge_idx)
                p_out = dataloader.dataset.scaler.inverse_transform(p_out)
                
                p_out = p_out.float().to(device)
                p_loss = loss_fn(p_out,r_out, y[:,:self.predict_len,:],x)
                loss = p_loss[1]
                loss.backward()
                optimizer.step()
                
                iter +=1
                if iter % 100 == 0:
                    cost = time.time()-start
                    start = time.time()
                    print('iter',iter,'time',cost)
                mse.append(p_loss[0].item())
                mae.append(p_loss[1].item())
            scheduler.step()
            train_mse_list.append(np.mean(mse))
            train_mae_list.append(np.mean(mae))
            if self.show_fig == True:
                print('train')
                plot_pred(dataloader.dataset.scaler.inverse_transform(x.cpu())[0,:,:2,0].T.tolist(),y.cpu()[0,:self.predict_len,:2].T.tolist(),p_out.cpu()[0,:,:2].T.tolist())
            
            val_loss = self.vali()
            val_mse_list.append(val_loss[0])
            val_mae_list.append(val_loss[1])
            # each epoch
            if self.show_fig == True:
                plot_loss(train_mse_list,train_mae_list,val_mse_list,val_mae_list)
            print('epoch ({} / {}) (train_loss:{:.8f},{:.8f})(val_loss:{:.8f},{:.8f})'.format(
                            i_epoch+1, epoch, 
                            np.mean(mse),np.mean(mae),val_loss[0],val_loss[1]), flush=True
                )
        
        if not os.path.exists(self.save_path+'/'+self.dataset):
            os.makedirs(self.save_path+'/'+self.dataset)
        plot_loss(train_mse_list,train_mae_list,val_mse_list,val_mae_list)
        plt.subplot(2,1,1)
        plt.plot(range(epoch), train_mse_list,range(epoch),val_mse_list)
        plt.title('mse')
        plt.ylabel('loss')
        plt.legend(['train','val'])
        plt.subplot(2,1,2)
        plt.plot(range(epoch), train_mae_list,range(epoch),val_mae_list)
        plt.title('mae')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train','val'])
        plt.savefig('loss.png')
        torch.save(model.state_dict(), self.save_path+f'/{self.dataset}/epoch{i_epoch+1}_{self.predict_len}.pt')

    def vali(self):
        self.model.eval()

        i = 0
        acu_loss = 0
        mse_loss_list=[]
        mae_loss_list=[]
        for i,(x, y,node_edge_index,res_edge_index) in enumerate(self.valiloader):
            b,t,n,_=x.shape
            node_edge_index,res_edge_index = node_edge_index.squeeze(1),res_edge_index.squeeze(1)
            x, y = [item.float().to(self.device) for item in [x, y]]
            with torch.no_grad():
                recon,predicted,attn_w1,attn_w2 = self.model(x, node_edge_index, res_edge_index)
                predicted = predicted.float().to(self.device)
                predicted = self.valiloader.dataset.scaler.inverse_transform(predicted)
                p_loss = loss_fn(predicted,recon, y[:,:self.predict_len,:],x)
                mse_loss_list.append(p_loss[0].item())
                mae_loss_list.append(p_loss[1].item())
        if self.show_fig == True:
            print('validation')
            plot_pred(self.valiloader.dataset.scaler.inverse_transform(x.cpu())[0,:,:2,0].T.tolist(),y.cpu()[0,:self.predict_len,:2].T.tolist(),predicted[0,:,:2].T.tolist())
        mse_loss = np.mean(mse_loss_list)
        mae_loss = np.mean(mae_loss_list)
        return mse_loss,mae_loss
    
    def test(self):
        print('###################testing################')
        device = self.device
        dataloader = self.testloader
        self.model.load_state_dict(torch.load(self.save_path+f'/{self.dataset}/epoch{self.pretrained_epochs}_{self.predict_len}.pt'))
        model = self.model
        if self.device == 'cuda':
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
        mse_loss_list=[]
        mae_loss_list=[]
        iter = 0
        for i,(x, y,node_edge_index,res_edge_index) in enumerate(dataloader):
            b,t,n,_=x.shape

            node_edge_index,res_edge_index = node_edge_index.squeeze(1),res_edge_index.squeeze(1)
            x, y = [item.to(device).float() for item in [x, y]]
            with torch.no_grad():
                recon,predicted,attn_w1,attn_w2 = model(x, node_edge_index, res_edge_index)
                predicted = predicted.float().to(device)
                predicted = dataloader.dataset.scaler.inverse_transform(predicted)
                p_loss = loss_fn(predicted,recon, y[:,0:self.predict_len,:],x)
                mse_loss_list.append(p_loss[0].item())
                mae_loss_list.append(p_loss[1].item())
                iter +=1
                if iter % 100 == 0:
                    plot_pred(dataloader.dataset.scaler.inverse_transform(x.cpu())[0,:,:3,0].T.tolist(),y.cpu()[0,0:self.predict_len,:3].T.tolist(),predicted[0,:,:3].T.tolist())
                    print('iter',iter)
        mse_loss = sum(mse_loss_list)/len(mse_loss_list)
        mae_loss = np.mean(mae_loss_list)
        #sup_para = df({'epoch':[self.pretrained_epochs],'lr':[self.lr],'dp':[self.dropout],'bs':[self.batch_size],'dataset':[self.dataset],
        #'win_s':[self.win_size],'predict_len':[self.predict_len],'emb_s':[self.emb_size]})
        #performance = df({'mse':[mse_loss],'mae':[mae_loss]})
        #excel_logs('log2.xlsx',sup_para,performance)
        print('mse:',mse_loss,'mae:',mae_loss)
        """
            for j in range(x.shape[0]+x.shape[2]):
                    test_recon_l_list.append([])
            x, y, labels, node_edge_index, res_edge_index = [item.to(device).float() for item in [x, y, labels, node_edge_index, res_edge_index]]
            
            with torch.no_grad():
                recon,predicted,attn_w1,attn_w2 = model(x, node_edge_index, res_edge_index)
                predicted = predicted.float().to(device)
                recon = recon.float().to(device)
                p_loss = loss_fn(predicted,recon, y,x)
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
        performence = self.get_score([test_predicted_list,test_ground_list,test_labels_list],test_recon_l_list)
        para = df({'epoch':[self.num_epochs],'lr':[self.lr],'dp':[self.dropout],'bs':[self.batch_size],'dataset':[self.dataset],
        'win_s':[self.win_size],'emb_s':[self.emb_size]})
        excel_logs('log.xlsx',para,performence)

    def get_score(self, test_result,test_recon):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores = get_full_err_scores(test_result)
        t_recon_ = [np.mean(l) for l in test_recon]
        t_recon = np.array(t_recon_).reshape(1,-1)
        #test_scores = test_scores + t_recon
        
        info = get_best_performance_data(test_scores, test_labels, topk=1) 
        performence = df([[info[0],info[1],info[2]]],columns=['F1 score','precision','recall'])

        print('=========================** Result **============================\n')

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')
        return performence
        """