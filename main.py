from ast import arg
from distutils.command.config import config
from email import parser
import os
import argparse
import torch

from detector import Detector

parser = argparse.ArgumentParser()

parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=int, default=0.2)
parser.add_argument('--batch_size', type=int, default=36)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dataset', type=str, default='msl')
parser.add_argument('--win_size', type=int, default=20)
parser.add_argument('--feature_num', type=int, default=27)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--gamma', type=int, default=0.5)
parser.add_argument('--save_path', type=str, default='result')

parser.add_argument('--em_type', type=str, default='rnn')
parser.add_argument('--emb_size', type=int, default=20)
parser.add_argument('--graph_type', type=str, default='full')
parser.add_argument('--gat_heads', type=int, default=1)

parser.add_argument('--recon_m', type=str, default='attn')
parser.add_argument('--forecast_m', type=str, default='mlp')

parser.add_argument('--attn_heads', type=int, default=1)

parser.add_argument('--mlp_layer', type=int, default=2)

config = parser.parse_args()
args = vars(config)

if args['device'] == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark=True

detector = Detector(args)