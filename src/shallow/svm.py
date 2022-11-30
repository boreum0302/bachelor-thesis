import numpy as np
import itertools
import pandas as pd

from ._base import split, standardize

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

def hyperparameter_optimize(X_train, y_train, X_valid, y_valid):
        
    max_auc = -1
    C, gamma = None, None
    
    for C_candidate, gamma_candidate in itertools.product([2**n for n in range(-5, 16, 2)], [2**n for n in range(-15, 4, 2)]):
        
        model = SVC(
            kernel='rbf', C=C_candidate, gamma=gamma_candidate
        )

        model.fit(X_train, y_train.ravel())

        auc = roc_auc_score(
            y_true=y_valid,
            y_score=model.decision_function(X_valid)
        )
        
        if auc > max_auc:
            max_auc = auc
            C, gamma = C_candidate, gamma_candidate
            
    return C, gamma


def save_results(dataset, path, seed=0):

    np.random.seed(seed)

    Dataset_col = []
    C_col = []
    gamma_col = []
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
            C, gamma = hyperparameter_optimize(X_train, y_train, X_valid, y_valid)

            # evaluate
            model = SVC(
                kernel='rbf', C=C, gamma=gamma
            )
            
            model.fit(X_train, y_train.ravel())
            
            auc = roc_auc_score(
                y_true=y_test,
                y_score=model.decision_function(X_test)
            )
            
            # save
            Dataset_col.append(data_name)
            C_col.append(C)
            gamma_col.append(gamma)
            AUC_col.append(auc)
            
            print('%-12s %02d %.4f' %(data_name, trial, auc))
                        
    tbl = pd.DataFrame(
        {'Dataset': Dataset_col,
        'C': C_col,
        'gamma': gamma_col,
        'AUC': AUC_col}
    )

    tbl.to_csv(path, index=False)