# https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py

import numpy as np
import pandas as pd
import easydict

from ._base import split, standardize

import torch

from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from easydict import EasyDict


##################
# Build Networks #
##################

class AutoEncoderNet(nn.Module):
    
    def __init__(self, x_dim, h_dims, rep_dim, bias=False):
        
        super(AutoEncoderNet, self).__init__()

        # encoder
        encoding_layers = []
        neurons = [x_dim, *h_dims]
        for idx in range(len(neurons) - 1):
            encoding_layers.append(nn.Linear(neurons[idx], neurons[idx + 1], bias=bias))
            encoding_layers.append(nn.BatchNorm1d(neurons[idx + 1], affine=bias))
            encoding_layers.append(nn.LeakyReLU())
            
        encoding_layers.append(nn.Linear(h_dims[-1], rep_dim, bias=bias))
        
        self.encoder = nn.Sequential(*encoding_layers)
        
        # decoder
        decoding_layers = []
        neurons = [rep_dim, *reversed(h_dims)]
        for idx in range(len(neurons) - 1):
            decoding_layers.append(nn.Linear(neurons[idx], neurons[idx + 1], bias=bias))
            decoding_layers.append(nn.BatchNorm1d(neurons[idx + 1], affine=bias))
            decoding_layers.append(nn.LeakyReLU())
            
        decoding_layers.append(nn.Linear(h_dims[0], x_dim, bias=bias))
        
        self.decoder = nn.Sequential(*decoding_layers)
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


class DeepSVDDNet(nn.Module):
    
    def __init__(self, x_dim, h_dims, rep_dim, bias=False):
        
        super(DeepSVDDNet, self).__init__()

        # encoder
        encoding_layers = []
        neurons = [x_dim, *h_dims]
        for idx in range(len(neurons) - 1):
            encoding_layers.append(nn.Linear(neurons[idx], neurons[idx + 1], bias=bias))
            encoding_layers.append(nn.BatchNorm1d(neurons[idx + 1], affine=bias))
            encoding_layers.append(nn.LeakyReLU())
            
        encoding_layers.append(nn.Linear(h_dims[-1], rep_dim, bias=bias))
        
        self.encoder = nn.Sequential(*encoding_layers)
        
    def forward(self, x):
        
        x = self.encoder(x)
        
        return x
    
    
###################
# Class Deep SVDD #
###################

class DeepSVDD():
    
    def __init__(self, args):
        
        self.args = args
    
    def pretrain(self, data):
        
        dataloader = DataLoader(data, batch_size=self.args.batch_size)
        
        pretrained_net = AutoEncoderNet(
            x_dim=self.args.x_dim, h_dims=self.args.h_dims, rep_dim=self.args.rep_dim
        )
        
        optimizer = torch.optim.Adam(
            pretrained_net.parameters(),
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        
        pretrained_net.train()
        
        for epoch in range(self.args.n_epochs):
                        
            for X in dataloader:
                
                X = X.to('cpu')
                
                optimizer.zero_grad()
                
                loss_fn = nn.MSELoss()
                loss = loss_fn(pretrained_net(X), X)
                loss.backward()
                
                optimizer.step()
        
        # initialize c
        pretrained_net.eval()
        Z = pretrained_net.encoder(data)
        c = torch.mean(Z, dim=0)
        c[(abs(c) < self.args.eps) & (c < 0)] = -self.args.eps
        c[(abs(c) < self.args.eps) & (c > 0)] = +self.args.eps
        
        # initialize W
        net = DeepSVDDNet(
            x_dim=self.args.x_dim, h_dims=self.args.h_dims, rep_dim=self.args.rep_dim
        ).to('cpu')
        net.load_state_dict(pretrained_net.state_dict(), strict=False)
        
        torch.save(
            {'c': c.cpu().data.numpy().tolist(), 'W': net.state_dict()},
            'pretrained.pth'
        )
        
    def train(self, data):
        
        dataloader = DataLoader(data, batch_size=self.args.batch_size)
        
        saved = torch.load('pretrained.pth')
        
        c = torch.Tensor(saved['c']).to('cpu')
        
        net = DeepSVDDNet(
            x_dim=self.args.x_dim, h_dims=self.args.h_dims, rep_dim=self.args.rep_dim
        ).to('cpu')
        net.load_state_dict(saved['W'])
        
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        
        net.train()
        
        for epoch in range(self.args.n_epochs):
                        
            for X in dataloader:
                
                X = X.to('cpu')
                
                optimizer.zero_grad()
                
                Z = net(X)
                loss = torch.mean(torch.sum((Z - c)**2, dim=1))
                loss.backward()
                
                optimizer.step()
                                
        self.net = net
        self.c = c
    
    def test(self, X, y):
        
        net = self.net
        c = self.c
        
        net.eval()
        
        scores = torch.sum((net(X) - c)**2, dim=1)        
        scores = scores.detach().numpy()
        
        auc = roc_auc_score(y_true=y, y_score=scores)
        
        return auc
    

def save_results(dataset, path, seed=0):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    Dataset_col = []
    AUC_col = []

    
    for data_name, data in dataset.items():
        
        # set arguments
        if data_name == 'arrhythmia':
            x_dim=274; h_dims=[128, 64]; rep_dim=32
        if data_name == 'cardio':
            x_dim=21; h_dims=[32, 16]; rep_dim=8
        if data_name == 'satellite':
            x_dim=36; h_dims=[32, 16]; rep_dim=8
        if data_name == 'satimage_2':
            x_dim=36; h_dims=[32, 16]; rep_dim=8
        if data_name == 'shuttle':
            x_dim=9; h_dims=[32, 16]; rep_dim=8
        if data_name == 'thyroid':
            x_dim=6; h_dims=[32, 16]; rep_dim=4

        args = EasyDict(
            {
                'x_dim': x_dim,
                'h_dims': h_dims,
                'rep_dim': rep_dim,
                'batch_size': 128,
                'n_epochs': 150,
                'lr': 1e-3,
                'weight_decay': 1e-6,
                'eps': 0.1
            }
        )
        
        # load data
        X, y = data['X'].astype('float32'), data['y'].astype('float32')
        
        for trial in range(1, 11):
            
            # split
            X_train, X_test, y_train, y_test = split(X, y)
            
            # standardize
            X_train = standardize(X_train)
            X_test = standardize(X_test)
            
            # covert to tensor
            X_train = torch.tensor(X_train)
            X_test = torch.tensor(X_test)
            
            # train and test
            trainer = DeepSVDD(args)
            trainer.pretrain(X_train)
            trainer.train(X_train)
            auc = trainer.test(X_test, y_test)
            
            # save
            Dataset_col.append(data_name)
            AUC_col.append(auc)
            
            print('%-12s %02d %.4f' %(data_name, trial, auc))
            
    tbl = pd.DataFrame(
        {'Dataset': Dataset_col,
        'AUC': AUC_col}
    )

    tbl.to_csv(path, index=False)

