# https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/optim/ae_trainer.py

import numpy as np
import pandas as pd

from ._base import split, standardize

import torch

from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


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


def create_net(data_name):
    
    if data_name == 'arrhythmia':
        return AutoEncoderNet(x_dim=274, h_dims=[128, 64], rep_dim=32)
    
    if data_name == 'cardio':
        return AutoEncoderNet(x_dim=21, h_dims=[32, 16], rep_dim=8)
        
    if data_name == 'satellite':
        return AutoEncoderNet(x_dim=36, h_dims=[32, 16], rep_dim=8)
    
    if data_name == 'satimage_2':
        return AutoEncoderNet(x_dim=36, h_dims=[32, 16], rep_dim=8)
    
    if data_name == 'shuttle':
        return AutoEncoderNet(x_dim=9, h_dims=[32, 16], rep_dim=8)
    
    if data_name == 'thyroid':
        return AutoEncoderNet(x_dim=6, h_dims=[32, 16], rep_dim=4)
    
    
def train(dataloader, net, n_epochs):
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)
    
    for epoch in range(n_epochs):
        
        for X in dataloader:
            
            X = X.to('cpu')
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(net(X), X)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
def save_results(dataset, path, seed=0):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    Dataset_col = []
    AUC_col = []
    
    batch_size = 128; n_epochs = 150;
    
    for data_name, data in dataset.items():
        
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
            
            # conver to dataloader
            X_train_loader = DataLoader(X_train, batch_size=batch_size)
            X_test_loader = DataLoader(X_test, batch_size=batch_size)
            
            # train
            net = create_net(data_name)
            train(dataloader=X_train_loader, net=net, n_epochs=n_epochs)
            
            # test
            net.eval()
            X_test_reconstructed = net(X_test)
            mse = nn.MSELoss(reduction='none')
            scores = torch.mean(mse(X_test_reconstructed, X_test), dim=1)
            auc = roc_auc_score(y_true=y_test, y_score=scores.detach().numpy())
            
            # save
            Dataset_col.append(data_name)
            AUC_col.append(auc)
            
            print('%-12s %02d %.4f' %(data_name, trial, auc))
            
    tbl = pd.DataFrame(
        {'Dataset': Dataset_col,
        'AUC': AUC_col}
    )

    tbl.to_csv(path, index=False)