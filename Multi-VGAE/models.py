"""
@Time: 2022/12/19 19:40
@Author: gorgeousdays@outlook.com
@File: model.py
@Summary: 
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def glorot_init(input_dim, output_dim):
    stdd = np.sqrt(2.0 / (input_dim + output_dim))

    w = nn.Parameter(torch.Tensor(input_dim, output_dim))
    return torch.nn.init.normal_(w, mean=0, std=stdd)


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


class GAE(nn.Module):
    def __init__(self, adj, n_users, device, p_dims, q_dims=None, dropout=0.5):
        super(GAE, self).__init__()
        self.n_users = n_users
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1]
            assert q_dims[-1] == p_dims[0]
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        adj = self._convert_sp_mat_to_sp_tensor(adj).to(device)
        input_dim, hidden1_dim, hidden2_dim = self.p_dims[2], self.p_dims[1], self.p_dims[0]
        self.embeddings = nn.Embedding(num_embeddings=adj.shape[0], embedding_dim=input_dim)
        nn.init.normal_(self.embeddings.weight, std=0.1)
        
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x:x)

        self.drop = nn.Dropout(dropout)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def encode(self):
        h = self.drop(self.embeddings.weight)
        # h = self.embeddings.weight
        hidden = self.base_gcn(h)
        z = self.gcn_mean(hidden)
        return z

    def decode(self, Z):
        return torch.sigmoid(torch.matmul(Z[:self.n_users], Z[self.n_users:].t()))
        # return torch.sigmoid(torch.matmul(Z, Z.t()))

    def forward(self):
        z = self.encode()
        pred = self.decode(z)
        return pred


class VGAE(nn.Module):

    def __init__(self, adj, n_users, device, p_dims, q_dims=None, dropout=0.2):
        super(VGAE, self).__init__()
        self.n_users = n_users
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1]
            assert q_dims[-1] == p_dims[0]
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        adj = self._convert_sp_mat_to_sp_tensor(adj).to(device)
        input_dim, hidden1_dim, hidden2_dim = self.p_dims[2], self.p_dims[1], self.p_dims[0]
        self.embeddings = nn.Embedding(num_embeddings=adj.shape[0], embedding_dim=input_dim)
        nn.init.normal_(self.embeddings.weight, std=0.1)

        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim * 2, adj, activation=lambda x:x)

        self.drop = nn.Dropout(dropout)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def encode(self):
        h = self.drop(self.embeddings.weight)
        # h = self.embeddings.weight
        hidden = self.base_gcn(h)
        h = self.gcn_mean(hidden)
        mu = h[:, :self.q_dims[-1]]
        logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, Z):
        return torch.matmul(Z[:self.n_users], Z[self.n_users:].t())

    def forward(self):
        mu, logvar = self.encode()
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class MultiVGAE(nn.Module):
    def __init__(self, adj_ui, adj_uu, n_users, device, p_dims, q_dims=None, dropout=0.2):
        super(MultiVGAE, self).__init__()
        self.n_users = n_users
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1]
            assert q_dims[-1] == p_dims[0]
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        input_dim, hidden1_dim, hidden2_dim = self.p_dims[2], self.p_dims[1], self.p_dims[0]

        adj_ui = self._convert_sp_mat_to_sp_tensor(adj_ui).to(device)
        adj_uu = self._convert_sp_mat_to_sp_tensor(adj_uu).to(device)

        self.embeddings = nn.Embedding(num_embeddings=adj_ui.shape[0], embedding_dim=input_dim)
        nn.init.normal_(self.embeddings.weight, std=0.1)

        self.base_gcn_ui = GraphConvSparse(input_dim, hidden1_dim, adj_ui)
        self.gcn_mean_ui = GraphConvSparse(hidden1_dim, hidden2_dim * 2, adj_ui, activation=lambda x: x)

        self.base_gcn_uu = GraphConvSparse(input_dim, hidden1_dim, adj_uu)
        self.gcn_mean_uu = GraphConvSparse(hidden1_dim, hidden2_dim * 2, adj_uu, activation=lambda x: x)

        self.drop = nn.Dropout(dropout)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def encode_ui(self):
        h = self.drop(self.embeddings.weight)
        # h = self.embeddings.weight
        hidden = self.base_gcn_ui(h)
        h = self.gcn_mean_ui(hidden)
        mu = h[:, :self.q_dims[-1]]
        logvar = h[:, self.q_dims[-1]:]
        return mu, logvar
   
    def encode_uu(self):
        h = self.drop(self.embeddings.weight[:self.n_users])
        # h = self.embeddings.weight
        hidden = self.base_gcn_uu(h)
        h = self.gcn_mean_uu(hidden)
        mu = h[:, :self.q_dims[-1]]
        logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_ui(self, Z):
        return torch.matmul(Z[:self.n_users], Z[self.n_users:].t())

    def decode_uu(self, Z):
        return torch.matmul(Z, Z.t())

    def forward(self):
        mu_ui, logvar_ui = self.encode_ui()
        z_ui = self.reparameterize(mu_ui, logvar_ui)

        mu_uu, logvar_uu = self.encode_uu()
        z_uu = self.reparameterize(mu_uu, logvar_uu)
        return self.decode_ui(z_ui), mu_ui, logvar_ui, self.decode_uu(z_uu), mu_uu, logvar_uu


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.drop = nn.Dropout(dropout)
        # self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def loss_function_multivae(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD

def loss_function_gae(recon_x, x):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))

    return BCE

def loss_function_vgae(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD

def loss_function_multivgae(recon_ui, ui, mu_ui, logvar_ui, recon_uu, uu, mu_uu, logvar_uu, anneal=1.0):
    BCE1 = -torch.mean(torch.sum(F.log_softmax(recon_ui, 1) * ui, -1))
    # BCE1 = F.mse_loss(recon_ui, ui)
    KLD1 = -0.5 * torch.mean(torch.sum(1 + logvar_ui - mu_ui.pow(2) - logvar_ui.exp(), dim=1))

    BCE2 = -torch.mean(torch.sum(F.log_softmax(recon_uu, 1) * uu, -1))
    # BCE2 = F.mse_loss(recon_uu, uu)
    KLD2 = -0.5 * torch.mean(torch.sum(1 + logvar_uu - mu_uu.pow(2) - logvar_uu.exp(), dim=1))
    # BCE3 = -torch.mean(torch.sum(F.log_softmax(recon_ui * )))
    anneal = 0.1
    return BCE1 + 0.01 * BCE2 + anneal * (KLD1 + KLD2)