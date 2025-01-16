"""
@Time: 2022/12/19 19:40
@Author: gorgeousdays@outlook.com
@File: main.py
@Summary:
"""

import numpy as np
import scipy.sparse as sp
import pandas as pd
import numpy as np
from scipy import sparse


def load_data(dataset):
    train_file = "data/" + dataset + '/' + dataset + ".train.txt"
    test_file = "data/" + dataset + '/' + dataset + ".test.txt"
    friendship_file = "data/" + dataset + "/friendship2.txt"

    n_items, n_users = 0, 0
    with open(train_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                n_items = max(n_items, max(items))
                n_users = max(n_users, uid)
    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                n_items = max(n_items, max(items))

    n_users += 1
    n_items += 1
    train_adj = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    test_adj = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    with open(train_file) as f_train:
        with open(test_file) as f_test:
            for l in f_train.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]
                for i in train_items:
                    train_adj[uid, i] = 1
            for l in f_test.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, test_items = items[0], items[1:]
                for i in test_items:
                    test_adj[uid, i] = 1

    train_adj_all = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    train_adj_all = train_adj_all.tolil()
    train_adj = train_adj.tolil()
    train_adj_all[:n_users, n_users:] = train_adj
    train_adj_all[n_users:, :n_users] = train_adj.T

    #   Load friendship adj
    friendship_adj = sp.dok_matrix((n_users, n_users), dtype=np.float32)
    with open(friendship_file) as f_friend:
        for l in f_friend.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')]
            uid, friends = items[0], items[1:]
            # friendship_adj[uid, uid] = 1 # 增加自连接 不然可能会出现nan值
            for i in friends:
                friendship_adj[uid, i] = 1

    return train_adj.tocsr(), test_adj.tocsr(), train_adj_all.tocsr(), n_users, n_items, friendship_adj.tocsr()
