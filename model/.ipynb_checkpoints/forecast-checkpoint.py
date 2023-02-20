import torch
from torch import nn

class MLPPredicter(nn.Module):
    def __init__(self, in_num,out_num ,layer_num,feature_num, inter_num=512,dropout=0):
        super(MLPPredicter, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, out_num))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Dropout(dropout))
                modules.append(nn.Linear( layer_in_num, inter_num ))
                #modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.LayerNorm([feature_num,inter_num]))
                modules.append(nn.LeakyReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out.permute(0,2,1)