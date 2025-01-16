"""
@Time: 2022/12/19 19:40
@Author: gorgeousdays@outlook.com
@File: main.py
@Summary: 
"""
# TODO: 激活函数试下prelu, leakyrelu
# TODO: loss function改为只计算非零值
# TODO: VGAE和GAE模型return的size为user-item矩阵的size 尝试改为邻接矩阵后再计算loss(跟adj的loss 便于后续注入friendship)
import torch
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
import time
import warnings
import pickle

warnings.filterwarnings('ignore')

import metric
from args import parser
from load_data import load_data
from models import MultiVAE, VGAE, GAE, MultiVGAE
from models import loss_function_multivae, loss_function_gae, loss_function_vgae, loss_function_multivgae

args = parser.parse_args()

# Set the random seed.
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device("cuda:1")
###############################################################################
# Load data
train_data, test_data, train_adj, n_users, n_items, uu_adj = load_data(args.dataset)
N = n_users
idxlist = list(range(n_users))


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

train_adj = preprocess_graph(train_adj)
uu_adj = preprocess_graph(uu_adj)

update_count = 0
###############################################################################
def naive_sparse2tensor(data):
   return torch.FloatTensor(data.toarray())

def train_all(model, optimizer):
    model.train()
    global update_count

    data = train_data
    data = naive_sparse2tensor(data).to(device)

    if args.total_anneal_steps > 0:
        anneal = min(args.anneal_cap,
                     1. * update_count / args.total_anneal_steps)
    else:
        anneal = args.anneal_cap
    if args.model == "MultiVAE":
        recon_batch, mu, logvar = model(data)
        loss = loss_function_multivae(recon_batch, data, mu, logvar, anneal)
    elif args.model == "GAE":
        recon_batch = model()
        loss = loss_function_gae(recon_batch, data)
    elif args.model == "VGAE":
        recon_batch, mu, logvar = model()
        loss = loss_function_vgae(recon_batch, data, mu, logvar, anneal)
    elif args.model == "MultiVGAE":
        uu_adj_data = naive_sparse2tensor(uu_adj).to(device)
        recon_batch_ui, mu_ui, logvar_ui, recon_batch_uu, mu_uu, logvar_uu = model()
        loss = loss_function_multivgae(recon_batch_ui, data, mu_ui, logvar_ui, recon_batch_uu, uu_adj_data, mu_uu, logvar_uu, anneal)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_count += 1

    return loss.item()


def train_batch(model, optimizer):
    model.train()
    train_loss = 0.0
    global update_count

    np.random.shuffle(idxlist)
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)

        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                            1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap

        recon_batch, mu, logvar = model(data)


        loss = loss_function_multivae(recon_batch, data, mu, logvar, anneal)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_count += 1

    return train_loss

def evaluate_all(model):
    model.eval()
    global update_count
    with torch.no_grad():
        data = train_data
        heldout_data = test_data

        data_tensor = naive_sparse2tensor(data).to(device)

        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                         1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap
        if args.model == "MultiVAE":
            recon_batch, mu, logvar = model(data_tensor)
            loss = loss_function_multivae(recon_batch, data_tensor, mu, logvar, anneal)
        if args.model == "GAE":
            recon_batch = model()
            loss = loss_function_gae(recon_batch, data_tensor)
        elif args.model == "VGAE":
            recon_batch, mu, logvar = model()
            loss = loss_function_vgae(recon_batch, data_tensor, mu, logvar, anneal)
        elif args.model == "MultiVGAE":
            uu_adj_data = naive_sparse2tensor(uu_adj).to(device)
            recon_batch_ui, mu_ui, logvar_ui, recon_batch_uu, mu_uu, logvar_uu = model()
            loss = loss_function_multivgae(recon_batch_ui, data_tensor, mu_ui, logvar_ui, recon_batch_uu, uu_adj_data, mu_uu,
                                           logvar_uu, anneal)
            recon_batch = recon_batch_ui

        recon_batch = recon_batch.cpu().numpy()
        recon_batch[data.nonzero()] = -np.inf
        n20 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
        r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)

    return loss, np.mean(r20), np.mean(n20)

def evaluate_batch(model):
    model.eval()
    total_loss = 0.0
    global update_count
    r20_list, n20_list = [], []
    with torch.no_grad():
        for start_idx in range(0, N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = train_data[idxlist[start_idx:end_idx]]
            heldout_data = test_data[idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap,
                               1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor)

            loss = loss_function_multivae(recon_batch, data_tensor, mu, logvar, anneal)
            total_loss += loss.item()

            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf
            n20 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
            r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)

            n20_list.append(n20)
            r20_list.append(r20)

    total_loss /= len(range(0, N, args.batch_size))
    n20_list = np.concatenate(n20_list)
    r20_list = np.concatenate(r20_list)

    return total_loss, np.mean(r20_list), np.mean(n20_list)


if __name__ == '__main__':

    p_dims = [200, 600, n_items]
    # p_dims = [200, 600, 1000]
    # Models
    if args.model == "MultiVAE":
        model = MultiVAE(p_dims).to(device)
    elif args.model == "GAE":
        model = GAE(train_adj, n_users, device, p_dims).to(device)
    elif args.model == "VGAE":
        model = VGAE(train_adj, n_users, device, p_dims).to(device)
    elif args.model == "MultiVGAE":
        model = MultiVGAE(train_adj, uu_adj, n_users, device, p_dims).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # import pickle
    #
    # with open('origin_embeddings.pkl', 'wb') as f:
    #     pickle.dump(model.embeddings, f)

    best_recall20 = 0
    best_res = []
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_all(model, optimizer)
        val_loss, recall20, ndcg20 = evaluate_all(model)
        print(recall20, ndcg20)
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:4.2f}s | train loss {:4.2f} |valid loss {:4.2f} | '
              'r20 {:5.4f} | n20 {:5.4f} '.format(
            epoch, time.time() - epoch_start_time, train_loss, val_loss,
            recall20, ndcg20))
        print('-' * 90)
        if recall20 > best_recall20:
            best_recall20 = recall20
            best_res = [recall20, ndcg20]
        # if epoch == 115:
        #     with open('learned_embeddings.pkl', 'wb') as f:
        #         pickle.dump(model.embeddings, f)

    print(best_res)
