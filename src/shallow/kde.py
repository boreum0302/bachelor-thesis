import numpy as np
import pandas as pd

from ._base import split, standardize

from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score

    
def hyperparameter_optimize(X_train, y_train, X_valid, y_valid):

    max_log_likelihood = -np.inf
    h = None
    
    for h_candidate in [np.sqrt(2)**n for n in range(1, 11)]:
        
        model = KernelDensity(kernel='gaussian', bandwidth=h_candidate).fit(X_train)
        log_likelihood = model.score(X_valid)

        if log_likelihood > max_log_likelihood:
            max_log_likelihood = log_likelihood
            h = h_candidate
            
    return h


def save_results(dataset, path, seed=0):

    np.random.seed(seed)

    Dataset_col = []
    h_col = []
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
            h = hyperparameter_optimize(X_train, y_train, X_valid, y_valid)

            # evaluate
            model = KernelDensity(kernel='gaussian', bandwidth=h).fit(X_train)
            
            auc = roc_auc_score(
                y_true=y_test,
                y_score=-model.score_samples(X_test)
            )
            
            # save
            Dataset_col.append(data_name)
            h_col.append(h)
            AUC_col.append(auc)
            
            print('%-12s %02d %.4f' %(data_name, trial, auc))
                        
    tbl = pd.DataFrame(
        {'Dataset': Dataset_col,
        'h': h_col,
        'AUC': AUC_col}
    )

    tbl.to_csv(path, index=False)