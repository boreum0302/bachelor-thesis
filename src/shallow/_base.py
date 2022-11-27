from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
    
def split(X, y, validation=False, ratio=[5, 1, 4]):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=ratio[2]/sum(ratio),
        random_state=np.random.randint(10**6)
    )
    
    if validation:
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, stratify=y_train, test_size=ratio[1]/(ratio[0] + ratio[1]),
            random_state=np.random.randint(10**6)
        )
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    else:
        
        return X_train, X_test, y_train, y_test


def standardize(X):
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    return X