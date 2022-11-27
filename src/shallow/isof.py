import numpy as np
import itertools
import pandas as pd

from ._base import split, standardize

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


def save_results(dataset, path, seed=0):

    np.random.seed(seed)

    Dataset_col = []
    AUC_col = []

    for data_name, data in dataset.items():
        
        X, y = data['X'].astype('float32'), data['y'].astype('float32')
            
        for trial in range(1, 11):
            
            # split
            X_train, X_test, y_train, y_test = split(X, y)

            # standardize
            X_train = standardize(X_train)
            X_test = standardize(X_test)

            # evaluate
            model = IsolationForest(
                n_estimators=100, max_samples=256, random_state=0
            )
            
            model.fit(X_train, y_train.ravel())
            
            auc = roc_auc_score(
                y_true=y_test,
                y_score=(-1)*model.decision_function(X_test).ravel()
            )
            
            # save
            Dataset_col.append(data_name)
            AUC_col.append(auc)
            
            print('%-12s %02d %.4f' %(data_name, trial, auc))
                        
    tbl = pd.DataFrame(
        {'Dataset': Dataset_col,
        'AUC': AUC_col}
    )

    tbl.to_csv(path, index=False)