import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


class GATv2Conv(MessagePassing):
    

    def __init__(self, in_channels: int,
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 share_weights: bool = False, use_gatv2:bool = True,
                 **kwargs):
        super(GATv2Conv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels  # 输入特征维度，即每个图节点的维度
        self.out_channels = out_channels  # 输出特征维度
        self.heads = heads  # multi-head策略
        self.concat = concat  # 表示multi-head输出后的多个特征向量的处理方法是否需要拼接
        self.negative_slope = negative_slope  # 采用leakyRELU的激活函数，x的负半平面斜率系数
        self.dropout = torch.nn.Dropout(dropout)
        self.add_self_loops = add_self_loops  # GAT要求加入自环，即每个节点要与自身连接
        self.share_weights = share_weights  # 共享权重
        # linear transformation, parametrized by a weight matrix
        # 如果共享则source和target的weight一样
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)
        # self-attention on the nodes—a shared attentional mechanism a
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index,   # x图节点，edge_index边对应的索引，二维，一维表示source，令一维表示target
                size = None, return_attention_weights = None):
       
        H, C = self.heads, self.out_channels
        if torch.is_tensor(x):  # tensor返回1，不是返回0
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        # 给edge_index加入自环的节点索引
        if self.add_self_loops:
            if torch.is_tensor(edge_index):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index,edge_attr=None)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, torch.SparseTensor):
                edge_index = torch.set_diag(edge_index)


        # propagate的方法，这是一个集成方法，调用其会依次调用message、aggregate、update方法。
        # 在source_to_target的方式下，message方法负责产生source node需要传出的信息，aggregate负责为target node收集来自source node的信息
        # 一般是max、add（default）等方法，GAT默认采用的是add方法，update用于更新表示。
        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, torch.Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, torch.SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    # 重构message方法
    def message(self, x_j: torch.Tensor, x_i: torch.Tensor,
                index: torch.Tensor,   # edge_index中的第二维，即target node索引
                size_i):
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index)
        self._alpha = alpha
        alpha = self.dropout(alpha)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)