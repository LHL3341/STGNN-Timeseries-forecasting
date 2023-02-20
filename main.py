from ast import arg
from email import parser
import os
import argparse
import torch
import numpy as np
from solver import Solver
import random

def randseed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument('--num_epochs', type=int, default=35)
parser.add_argument('--pretrained_epochs',type = int,default=35)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--dp_mlp', type=int, default=0)
parser.add_argument('--dp_gat', type=int, default=0)
parser.add_argument('--dp_tcn', type=int, default=0.2)

parser.add_argument('--win_size', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dataset', type=str, default='nyc-bike')
parser.add_argument('--sr', type=bool, default=False)
parser.add_argument('--predict_len', type=int, default=12)#3,6,12,24
parser.add_argument('--feature_num', type=int, default=250)
parser.add_argument('--channels', type=list, default=[1,32,32,24,12])
#parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--gamma', type=int, default=0.5)
parser.add_argument('--save_path', type=str, default='result')

parser.add_argument('--em_type', type=str, default='tcn')
parser.add_argument('--emb_size', type=int, default=12)
parser.add_argument('--graph_type', type=str, default='full')
parser.add_argument('--gat_heads', type=int, default=1)

parser.add_argument('--recon_m', type=str, default='attn')
parser.add_argument('--forecast_m', type=str, default='mlp')

parser.add_argument('--attn_heads', type=int, default=4)

parser.add_argument('--mlp_layer', type=int, default=3)

parser.add_argument('--show_fig', type=bool, default=False)

config = parser.parse_args(args=[])
args = vars(config)

randseed(101)
if args['device'] == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark=True
detector = Solver(args)
if args['mode']=='train':
    detector.train()
else:
    detector.test()