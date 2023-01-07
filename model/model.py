import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from .embedding import TemporalConvNet
from .GATv2 import GATv2Conv
from .recon import AttentionDecoder
from .forecast import MLPPredicter
from utils.graph_structure import get_batch_edge_index


class Model(nn.Module):
    DEFAULTS = {}           
    def __init__(self,config) -> None:
        super(Model,self).__init__()
        self.__dict__.update(Model.DEFAULTS, **config)
        if self.mode == 'test':
            self.batch_size = 1
        self.gat = GATv2Conv(self.emb_size,self.emb_size,self.gat_heads)
        if self.em_type == 'rnn':
            self.embedding =nn.RNN(self.win_size,self.emb_size)
        elif self.em_type == 'tcn':
            self.embedding = TemporalConvNet(self.feature_num,[self.win_size,self.win_size])
        if self.recon_m == 'attn':
            self.r = AttentionDecoder(self.emb_size,self.attn_heads)
        if self.forecast_m =='mlp':
            self.f = MLPPredicter(self.emb_size,self.predict_len,self.mlp_layer)
    def forward(self,x,node_edge_idx,res_edge_idx):
        x = x.permute(0,2,1)
        y,_ = self.embedding(x)
        y = y.view(-1,self.emb_size)
        batch_size=node_edge_idx.shape[0]
        node_edge_idx = get_batch_edge_index(node_edge_idx,batch_size,self.feature_num)
        res_edge_idx = get_batch_edge_index(res_edge_idx,batch_size,self.feature_num)
        z1,attn_w1 = self.gat(y,node_edge_idx,return_attention_weights=True)
        z2,attn_w2 = self.gat(y,res_edge_idx,return_attention_weights=True)
        z=z1+z2
        z= z.view(-1,self.feature_num,self.emb_size)
        r = self.r(z)
        forecast = self.f(z)
        #recon = nn.Linear(self.emb_size,self.win_size)(r).permute(0,2,1)
        return _,forecast,attn_w1,attn_w2