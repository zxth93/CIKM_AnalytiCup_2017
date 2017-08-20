#encoding=utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise

# gets some time slice
def slice_t(train_df, time_sum, time_slice, m, n, h):

    train_np = train_df.reshape((time_sum, h, m, n))
    slice = train_np[time_sum-time_slice:]
    train = slice.reshape((time_slice*h*m*n))
    return train

# gets some height slice
def slice_h(arr, time, m, n, h, asd):

    train = np.zeros((time, h, m*n))

    for i in range(time):
        for j in range(h):
            train[i, j] = arr[i, j + asd]

    train = train.reshape((time, h, m, n))
    return train

# cleans trainning data,return data index
def pre_train(train_df, test_df, train_add, test_add):

    train = train_df.values[:,1:-1]
    t = train_add.values[:,1:-1]
    train = np.hstack((train, t))

    dtest = test_df.values[:,1:]
    tA = test_add.values[:,1:]
    dtest = np.hstack((dtest, tA))

    cor_distance = pairwise.pairwise_distances(dtest, train)

    resultset = set()
    for tmp in cor_distance:
        index = np.argsort(tmp)
        for i in range(10):
            resultset.add(index[i])

    index = []
    for i in resultset:
        index.append(i)

    return index

