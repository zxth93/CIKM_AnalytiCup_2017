#encoding=utf-8
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

# produces xgboost model
def xgb_train(train_df, test_df, mode, params,num_boost_round,early_stopping):

    if mode == "train":

        train = train_df.values[:,1:-1]
        train_target = train_df.values[:,-1]

        # 5-fold
        kf = KFold(n_splits=5, shuffle=True)
        trainEorror = 0
        error = 0

        for train_index, valid_index in kf.split(train):
            x_train, x_valid = train[train_index], train[valid_index]
            y_train, y_valid = train_target[train_index], train_target[valid_index]

            dtrain = xgb.DMatrix(x_train, y_train)
            dvalid = xgb.DMatrix(x_valid, y_valid)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                            early_stopping_rounds=early_stopping, verbose_eval=True)

            print("validating")

            tranHat = gbm.predict(xgb.DMatrix(x_train))
            trainEorror += rmsep(y_train, tranHat)
            yhat = gbm.predict(xgb.DMatrix(x_valid))
            error += rmsep(y_valid, yhat)

        print('rmse:{:.6f}'.format(error/5.0))

    else:

        train = train_df.values[:,1:-1]
        train_target = train_df.values[:,-1]

        kf = KFold(n_splits=5, shuffle=True)

        result = np.zeros(2000)

        dtest = test_df.values[:,1:]
        dtest = xgb.DMatrix(dtest)

        for train_index, valid_index in kf.split(train):
            x_train, x_valid = train[train_index], train[valid_index]
            y_train, y_valid = train_target[train_index], train_target[valid_index]

            dtrain = xgb.DMatrix(x_train, y_train)
            watchlist = [(dtrain, 'train')]
            gbm = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                            evals=watchlist, early_stopping_rounds=early_stopping, )

            result += gbm.predict(dtest)

        result = result/5.0
        return result 

# chooses parameter set
def paramset(type):

    if type==1:
        return 0.1,10,5,1,0.8
    else:
        return 0.02,43,5,0.8,0.8


# trains single xgboost model
def xgboosttrain(train_df,test_df,type):

    eta,max_depth,min_child_weight,subsample,colsample_bytree=paramset(type)
#    print(eta,max_depth,min_child_weight,subsample,colsample_bytree)

    params = {"objective":"reg:linear",
            "booster":"gbtree",
            "eta":eta,
            "max_depth":max_depth,
            "min_child_weight":min_child_weight,
            "subsample":subsample,
            "colsample_bytree":colsample_bytree,
            "silent":0,
            "seed":200
          }
    num_boost_round = 1000
    early_stopping_rounds = 100

    result =  xgb_train(train_df,test_df, 'trai', params, num_boost_round, early_stopping_rounds)
    return result

# trians multi-xgboost model
def xgbmodeltrain(train_df,test_df):

    result1=xgboosttrain(train_df,test_df,1)
    result2=xgboosttrain(train_df,test_df,2)
    result=result1*0.5+result2*0.5+1
    return result

