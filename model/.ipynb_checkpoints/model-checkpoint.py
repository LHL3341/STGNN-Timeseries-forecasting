import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from .embedding import TemporalConvNet
from .GATv2 import GATv2Conv
from .tcn import dilated_inception
from .recon import AttentionDecoder
from .forecast import MLPPredicter
from utils.graph_structure import get_batch_edge_index,get_batch_index
import time


class Model(nn.Module):
    DEFAULTS = {}           
    def __init__(self,config) -> None:
        super(Model,self).__init__()
        self.__dict__.update(Model.DEFAULTS, **config)
        if self.mode == 'test':
            self.batch_size = 1
        self.gat = GATv2Conv(self.channels[3],self.channels[4],self.gat_heads,dropout = self.dp_gat)
        print(self.gat._modules)
        #self.gat1 = GATv2Conv(self.channels[3],self.channels[4],self.gat_heads,dropout = self.dp_gat)
        #if self.em_type == 'rnn':
            #self.embedding =nn.RNN(self.win_size,self.emb_size)
        if self.em_type == 'tcn':
            self.tcn = TemporalConvNet(2,[64,256,64],self.feature_num,dropout=self.dp_tcn)
            """
            self.tcn = nn.Sequential(dilated_inception(self.channels[0],self.channels[1]),nn.ReLU(),nn.Dropout(self.dp_tcn),nn.LayerNorm([self.feature_num,13]),
                   dilated_inception(self.channels[1],self.channels[2]),nn.ReLU(),nn.Dropout(self.dp_tcn),nn.LayerNorm([self.feature_num,7]),
                   dilated_inception(self.channels[2],self.channels[3]),nn.ReLU(),nn.Dropout(self.dp_tcn),nn.LayerNorm([self.feature_num,1]))
            """
        #if self.recon_m == 'attn':
            #self.r = AttentionDecoder(self.emb_size,self.attn_heads)
        if self.forecast_m =='mlp':
            self.f = MLPPredicter(self.channels[4],self.predict_len,self.mlp_layer,self.feature_num,dropout=self.dp_mlp)
        nodenum = self.feature_num
        
        i_idx = torch.arange(0, nodenum).T.unsqueeze(1).repeat(1, nodenum).flatten().to(self.device).unsqueeze(0)
        j_idx = torch.arange(0, nodenum).repeat(1, nodenum).to(self.device)
        self.node_edge_idx = torch.cat((i_idx,j_idx),dim=0)
        self.layernorm = nn.LayerNorm([self.feature_num,self.channels[4]])
    def forward(self,x,node_edge_idx,res_edge_idx):
        batch_size=node_edge_idx.shape[0]
        if self.em_type == 'rnn':
            x = x.squeeze(-1)
            x = x.permute(0,2,1).contiguous()
            
            y,_ = self.embedding(x)
            
            y = y.reshape(-1,self.emb_size).contiguous()
        #t1 = time.time()
        if self.em_type == 'tcn':
            x = x.permute(0,3,2,1).contiguous()
            #x=nn.functional.pad(x,(19-12,0,0,0))
            #x = x.squeeze(-1).permute(0,2,1)
            y,_ = self.tcn(x)
            
            #y = y.permute(0,3,2,1).reshape(-1,y.shape[1])
            y = y.reshape(-1,y.shape[3]).contiguous()

        #print(y.shape)
        #t2 = time.time()
        
        node_edge_idx = get_batch_index(self.node_edge_idx,batch_size)
        
        #res_edge_idx = get_batch_edge_index(res_edge_idx,batch_size,self.feature_num)
        
        #print(node_edge_idx.shape)
        #node_edge_idx,res_edge_idx = [item.to(self.device) for item in [node_edge_idx,res_edge_idx]]
        
        #t3 = time.time()
        #z1,attn_w1 = self.gat(y,node_edge_idx,return_attention_weights=True)
        #t4 = time.time()
        #z2,attn_w2 = self.gat1(y,res_edge_idx,return_attention_weights=True)
        #z=z1+z2
        z=y
        z= z.view(-1,self.feature_num,self.channels[4]).contiguous()
        forecast = self.f(z)#self.layernorm(z))
        #t5 = time.time()
        #print(t2-t1,t3-t2,t4-t3,t5-t4)
        #recon = nn.Linear(self.emb_size,self.win_size)(r).permute(0,2,1)
        return 0,forecast,0,0