import pandas as pd
import numpy as np
import math
from keras.models import load_model
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization
from keras.layers import GRU, Bidirectional, TimeDistributed
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_squared_error
from keras.layers.pooling import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from keras.regularizers import L1L2

np.random.seed(28)
set_random_seed(28)


def BiGRU(X_train, y_train, X_test, y_test, gru_units, dense_units, input_shape, \
           batch_size, epochs, drop_out, patience):

    model = Sequential()

    reg = L1L2(l1=0.2, l2=0.2)

    model.add(Bidirectional(GRU(units = gru_units, dropout= drop_out, activation='relu', recurrent_regularizer = reg,
                                return_sequences = True),
                                input_shape = input_shape,
                                merge_mode="concat"))

    model.add(BatchNormalization())

    model.add(TimeDistributed(Dense(dense_units, activation='relu')))
    model.add(BatchNormalization())

    model.add(Bidirectional(GRU(units = gru_units, dropout= drop_out, activation='relu', recurrent_regularizer=reg,
                                    return_sequences = True),
                             merge_mode="concat"))

    model.add(BatchNormalization())

    model.add(Dense(units=1))

    model.add(GlobalAveragePooling1D())

    print(model.summary())

    early_stopping = EarlyStopping(monitor="val_loss", patience = patience)

    model.compile(loss='mse', optimizer= 'adam')

    history_callback = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,\
              verbose=2, callbacks=[early_stopping], validation_data=[X_test, y_test], shuffle = True)

    return model, history_callback

def read_data(train_data, test_data):

    X = train_data.iloc[:, 1:-1].values
    y = train_data.iloc[:, -1].values

    tX = test_data.iloc[:, 1:].values

    X = X.reshape(-1, 15, 4, 10, 4)
    tX = tX.reshape(-1, 15, 4, 10, 4)

    #only take the second level for input
    X = X[:, :, 1:2, :, :]
    tX = tX[:, :, 1:2, :, :]
    X = X.reshape(-1, 15, 40)
    tX = tX.reshape(-1, 15, 40)
    return X, y, tX

def normalization(X_train, X_test, tX):
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    tX_shape = tX.shape

    X_train = X_train.reshape((X_train_shape[0], -1))
    X_test = X_test.reshape((X_test_shape[0], -1))
    tX = tX.reshape((tX_shape[0], -1))

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    tX = X_scaler.transform(tX)

    X_train = X_train.reshape(X_train_shape)
    X_test = X_test.reshape(X_test_shape)
    tX = tX.reshape(tX_shape)

    return X_train, X_test, tX


def BiGRU_train(train_data, test_data, error_sort, train_mode):

    #basic configuration
    batch_size = 512
    epochs = 28
    drop_out = 0.1
    clean_rate = 0.5
    patience = 5
    gru_units = 128
    dense_units = 32

    #read data
    print("#read data:")
    X, y, tX = read_data(train_data, test_data)

    #outliers clean
    clean_data = error_sort[0: int(clean_rate*len(error_sort))]
    clean_data = np.array(clean_data, dtype = np.int32)
    clean_data = np.sort(clean_data)

    train_valid_split_point = clean_data[int(len(error_sort)*clean_rate*0.9)]

    X_valid = X[train_valid_split_point:]
    y_valid = y[train_valid_split_point:]

    X = X[clean_data]
    y = y[clean_data]

    #train valid split
    slice_point = int(0.9*X.shape[0])
    X_train, X_test, y_train, y_test = \
            X[0:slice_point], X_valid, \
            y[0:slice_point], y_valid

    #shuffle the train data
    random_sort = np.random.choice(list(range(X_train.shape[0])), size = X_train.shape[0],\
                                   replace = False)
    X_train = X_train[random_sort]
    y_train = y_train[random_sort]

    #normalization
    print("#normalization:")
    X_train, X_test, tX = normalization(X_train, X_test, tX)

    #train BiGRU
    print('#train BiGRU:')
    if(train_mode=='online'):
        model, loss_history = BiGRU(X_train, y_train, X_test, y_test, gru_units = gru_units, dense_units=dense_units,\
                                    input_shape = (15, 40),\
                                    batch_size = batch_size, epochs = epochs, drop_out = drop_out, \
                                    patience = patience)
    else:
        # load the pre-trained model
        print('#load the pre-trained model:')
        model = load_model('/home/Team4/Team4/model/checkpoint-27-163.96.hdf5')

    #calculate root mean squared error
    trainPredict = model.predict(X_train)
    trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
    print('Train Score: %.5f RMSE' % (trainScore))

    testPredict = model.predict(X_test)
    testScore = math.sqrt(mean_squared_error(y_test, testPredict))
    print('Test Score: %.5f RMSE' % (testScore))

    #predict testB
    tX_predict =  model.predict(tX)
    tX_predict[tX_predict < 0] = 0
    tX_predict = tX_predict + 2

    return tX_predict
    # np.savetxt("submit_BiGRU_testB.csv", tX_predict)


