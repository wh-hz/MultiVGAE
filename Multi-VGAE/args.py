"""
@Time: 2022/12/23 14:46
@Author: gorgeousdays@outlook.com
@File: args.py
@Summary: 
"""
import argparse

parser = argparse.ArgumentParser(description='PyTorch Multi-VGAE')
parser.add_argument('--model', type=str, default='MultiVAE',
                    help='selected model, MultiVAE, GAE, VGAE, MultiVGAE')
parser.add_argument('--dataset', type=str, default='yelp2018',
                    help='Movielens-20m dataset location')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--seed', type=int, default=2022,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
# parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#                     help='report interval')

# parser.add_argument('--save', type=str, default='model.pt',
#                     help='path to save the final model')