import sys
sys.path.append('..')
import rfmodel as rf
import pandas as pd
import numpy as np
from dataprocess import FeatureSelect as fs
from dataprocess import data_process8 as dp
from dataprocess import generate_percentile as gp
import xgbmodel as xgbm
import bigrumodel as bigru


def check_code(mode, gru_mode):

    if(mode == 'simple'):
        train_df = pd.read_csv('/home/Team4/Team4/dataset/train_percentile.csv')
        test_df = pd.read_csv('/home/Team4/Team4/dataset/testB_percentile.csv')
        train_add = pd.read_csv('/home/Team4/Team4/dataset/train_old_wind_4240.csv')
        testA_add = pd.read_csv('/home/Team4/Team4/dataset/testB_old_wind_4240.csv')
        train_1ave8extend = pd.read_csv('/home/Team4/Team4/dataset/train_new_wind_1ave_8extend.csv')
        test_1ave = pd.read_csv('/home/Team4/Team4/dataset/testB_new_wind_1ave_8extend.csv')
    else:
        trainfile = '/home/Team4/CIKM2017/train.txt'
        testBfile = '/home/Team4/CIKM2017/testB.txt'
        #生成训练集数据,老的风
        train_add = dp.dataprocess(trainfile, data_type='train', windversion='old')
        #生成测试集B数据,老的风
        testA_add = dp.dataprocess(testBfile, data_type='testB', windversion='old')
        #生成训练集数据,1ave8extend
        train_1ave8extend = dp.dataprocess(trainfile, data_type='train', windversion='new')
        #生成测试集B数据,1ave
        test_1ave = dp.dataprocess(testBfile, data_type='testB', windversion='new')
        #生成训练集数据
        train_df = gp.data_process(trainfile, data_type='train')
        #生成测试集B数据
        test_df = gp.data_process(testBfile, data_type='testB')


    print('#data process has been done')

    result_xgb = xgbm.xgbmodeltrain(train_1ave8extend, test_1ave)

    print('#xgb model has been done')

    index = fs.pre_train(train_df=train_df, test_df=test_df, train_add=train_add, test_add=testA_add)

    valid = rf.rf_model(train_df, test_df, 'train', train_add, testA_add, ne=100)

    ne = 1100
    result_rf = rf.rf_model(train_df, test_df, 'trai', train_add, testA_add, ne, index=index)

    print('#rf model has been done')

    result_bigru = bigru.BiGRU_train(train_df, test_df, valid, gru_mode).reshape(2000)

    print('#bigru model has been done')

    ensemble = (result_xgb+result_rf+result_bigru)/3.0

    np.savetxt("/home/Team4/Team4/result/submit_Team4.csv", ensemble)

#check_code('simple', 'online')

check_code('all','no')
