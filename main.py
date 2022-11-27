# http://odds.cs.stonybrook.edu/

from scipy import io
import numpy as np
import pandas as pd

arrhythmia = io.loadmat('./data/arrhythmia.mat')
cardio = io.loadmat('./data/cardio.mat')
satellite = io.loadmat('./data/satellite.mat')
satimage_2 = io.loadmat('./data/satimage_2.mat')
shuttle = io.loadmat('./data/shuttle.mat')
thyroid = io.loadmat('./data/thyroid.mat')

from sklearn.model_selection import train_test_split

X = shuttle['X']
y = shuttle['y']

X_, X, y_, y = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

shuttle['X'] = X
shuttle['y'] = y

dataset = {
    'arrhythmia': arrhythmia,
     'cardio': cardio,
     'satellite': satellite,
     'satimage_2': satimage_2,
     'shuttle': shuttle,
     'thyroid': thyroid
}

Dataset_col = ['arrhythmia', 'cardio', 'satellite', 'satimage_2', 'shuttle', 'thyroid']
N_col = [data['X'].shape[0] for data in dataset.values()]
D_col = [data['X'].shape[1] for data in dataset.values()]
outliers_col = [np.mean(data['y'])*100 for data in dataset.values()]

from src.shallow import svm, rf, oc_svm, ssad, kde, isof
from src.deep import ae, deep_svdd

svm.save_results(dataset, './results/svm.csv', seed=0)
rf.save_results(dataset, './results/rf.csv', seed=0)
oc_svm.save_results(dataset, './results/ocsvm.csv', seed=0)
ssad.save_results(dataset, './results/ssad.csv', seed=0)
kde.save_results(dataset, './results/kde.csv', seed=0)
isof.save_results(dataset, './results/isof.csv', seed=0)
ae.save_results(dataset, './results/ae.csv', seed=0)
deep_svdd.save_results(dataset, './results/deep_svdd.csv', seed=0)

