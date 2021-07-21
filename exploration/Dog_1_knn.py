#!/usr/bin/env python3

import preprocess
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn import metrics

# load input files
rootdir = '/ECShome/si1u19/data_copy/Dog_1/Dog_1/'
interictal_count = preprocess.count_interictal_files(rootdir)
interictal_files = preprocess.get_interictal_files(rootdir, interictal_count)
preictal_files = preprocess.get_preictal_files(rootdir)
print('Interictal file count:', len(interictal_files))
print('Preictal file count:', len(preictal_files))

# target vector
interictal_y = np.zeros(len(interictal_files))
preictal_y = np.ones(len(preictal_files))
y = np.concatenate((interictal_y, preictal_y), axis=0)

# input vector
X_list = []
X_list.extend(interictal_files)
X_list.extend(preictal_files)
X = np.array(X_list)
print(type(X))

# summarize class distribution (0=interictal, 1=pre-ictal)
print(Counter(y))

# define oversampling strategy due to class imbalance
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X.reshape(-1,1), y)
print(Counter(y_over))

X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)

# convert to z-score and flatten training set
X_train_EEG = np.zeros((643, 3836256))
for i in tqdm(range(len(X_train))):
    filepath = rootdir + X_train[i].item()
    mat = sio.loadmat(filepath)
    sigbuf = preprocess.get_sig(mat).astype('float32')
    mean, std = sigbuf.mean(), sigbuf.std()
    sigbuf = (sigbuf - mean) / std # normalise
    X_train_EEG[i,:] = sigbuf.flatten()


# convert to z-score and flatten test set
X_test_EEG = np.zeros((317, 3836256))
for i in tqdm(range(len(X_test))):
    filepath = rootdir + X_test[i].item()
    mat = sio.loadmat(filepath)
    sigbuf = preprocess.get_sig(mat).astype('float32')
    mean, std = sigbuf.mean(), sigbuf.std()
    sigbuf = (sigbuf - mean) / std # normalise
    X_test_EEG[i,:] = sigbuf.flatten()

k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_EEG, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test_EEG)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Done.")
