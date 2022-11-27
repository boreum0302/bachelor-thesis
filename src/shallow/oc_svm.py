import numpy as np
import pandas as pd
import itertools

from ._base import split, standardize

from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score


def hyperparameter_optimize(X_train, y_train, X_valid, y_valid):
        
    max_auc = -1
    gamma, nu = None, None
    
    for gamma_candidate, nu_candidate in itertools.product([2**n for n in range(-7, 3)], [0.01, 0.05, 0.1, 0.2, 0.5]):
    
        model = OneClassSVM(
            kernel='rbf',
            gamma=gamma_candidate,
            nu=nu_candidate
        )
            
        model.fit(X_train, y_train.ravel())

        max_ = max(model.score_samples(X_valid))
        auc = roc_auc_score(
            y_true=y_valid,
            y_score=max_ - model.score_samples(X_valid)
        )
        
        if auc > max_auc:
            max_auc = auc
            gamma, nu = gamma_candidate, nu_candidate
            
    return gamma, nu


def save_results(dataset, path, seed=0):

    np.random.seed(seed)

    Dataset_col = []
    gamma_col = []
    nu_col = []
    AUC_col = []

    for data_name, data in dataset.items():
        
        X, y = data['X'].astype('float32'), data['y'].astype('float32')
        
        for trial in range(1, 11):

            # split
            X_train, X_valid, X_test, y_train, y_valid, y_test = split(X, y, validation=True)

            # standardize
            X_train = standardize(X_train)
            X_valid = standardize(X_valid)
            X_test = standardize(X_test)

            # hyperparameter optimize
            gamma, nu = hyperparameter_optimize(X_train, y_train, X_valid, y_valid)

            # evaluate
            model = OneClassSVM(
                kernel='rbf',
                gamma=gamma,
                nu=nu
            )
            
            model.fit(X_train, y_train.ravel())
            
            max_ = max(model.score_samples(X_test))
            auc = roc_auc_score(
                y_true=y_test,
                y_score=max_ - model.score_samples(X_test)
            )
            
            # save
            Dataset_col.append(data_name)
            gamma_col.append(gamma)
            nu_col.append(nu)
            AUC_col.append(auc)
            
            print('%-12s %02d %.4f' %(data_name, trial, auc))
                        
    tbl = pd.DataFrame(
        {'Dataset': Dataset_col,
        'gamma': gamma_col,
        'nu': nu_col,
        'AUC': AUC_col}
    )

    tbl.to_csv(path, index=False)