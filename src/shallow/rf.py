import numpy as np
import pandas as pd

from ._base import split, standardize

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def hyperparameter_optimize(X_train, y_train, X_valid, y_valid):

    max_auc = -1
    max_features = None
    
    for max_features_candidate in np.linspace(1, X_train.shape[1], 10, dtype=int):
        
        model = RandomForestClassifier(
            max_features=max_features_candidate,
            criterion='entropy', random_state=0
        )
        
        model.fit(X_train, y_train.ravel())

        auc = roc_auc_score(
            y_true=y_valid,
            y_score=model.predict_proba(X_valid)[:, 1]
        )
        
        if auc > max_auc:
            max_auc = auc
            max_features = max_features_candidate
            
    return max_features


def save_results(dataset, path, seed=0):

    np.random.seed(seed)

    Dataset_col = []
    max_features_col = []
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
            max_features = hyperparameter_optimize(X_train, y_train, X_valid, y_valid)

            # evaluate
            model = RandomForestClassifier(
                max_features=max_features,
                criterion='entropy', random_state=0
            )
            
            model.fit(X_train, y_train.ravel())
            
            auc = roc_auc_score(
                y_true=y_test,
                y_score=model.predict_proba(X_test)[:, 1]
            )
            
            # save
            Dataset_col.append(data_name)
            max_features_col.append(max_features)
            AUC_col.append(auc)
            
            print('%-12s %02d %.4f' %(data_name, trial, auc))
                        
    tbl = pd.DataFrame(
        {'Dataset': Dataset_col,
         'max_features': max_features_col,
         'AUC': AUC_col}
    )

    tbl.to_csv(path, index=False)