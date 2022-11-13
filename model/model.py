import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from model.embedding import TemporalConvNet
from model.GATv2 import GATv2Conv
from model.recon import AttentionDecoder
from model.forecast import MLPPredicter
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
            self.r = AttentionDecoder(self.feature_num,self.attn_heads)
        if self.forecast_m =='mlp':
            self.f = MLPPredicter(self.emb_size,self.mlp_layer)
    def forward(self,x,edge_idx):
        x = x.permute(0,2,1)
        y,_ = self.embedding(x)
        y = y.view(-1,self.emb_size)
        edge_idx = get_batch_edge_index(edge_idx[0,:,:],edge_idx.shape[0],self.emb_size)
        z,attn_w = self.gat(y,edge_idx,return_attention_weights=True)
        z = z.view(-1,self.emb_size,self.feature_num)
        recon = self.r(z)
        forecast = self.f(z.permute(0,2,1))
        return recon,forecast,attn_w