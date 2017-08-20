#encoding=utf-8
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

# produces random forest model
def rf_model(train_df, test_df, mode, train_add,test_add, ne, index=0):

    if mode == "train":

        train = train_df.values[:,1:-1]
        train_target = train_df.values[:,-1]

        t = train_add.values[:,1:-1]
        train = np.hstack((train, t))

        rf = RandomForestRegressor(n_estimators=ne, verbose=2, n_jobs=-1)
        rf.fit(train, train_target)

        trainHat = rf.predict(train)
        valid = np.argsort(np.abs((trainHat - train_target)))

        return valid

    else:

        train_df = train_df.values[index]
        train = train_df[:,1:-1]
        train_target = train_df[:,-1]

        train_add = train_add.values[index]
        t = train_add[:,1:-1]

        #print(t.shape)
        train = np.hstack((train, t))

        kf = KFold(n_splits=2, shuffle=True)

        dtest = test_df.values[:,1:]
        tA = test_add.values[:,1:]
        dtest = np.hstack((dtest, tA))

        result = np.zeros(2000)

        for train_index, valid_index in kf.split(train):
            x_train, x_valid = train[train_index], train[valid_index]
            y_train, y_valid = train_target[train_index], train_target[valid_index]
            rf = RandomForestRegressor(n_estimators=ne, verbose=2, n_jobs=-1)
            rf.fit(x_train, y_train)
            result += rf.predict(dtest)

        result_rf = result/2.0 + 2

        return result_rf

