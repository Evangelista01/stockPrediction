# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 21:36:14 2018

@author: Evangelista
"""

import data_downloader as dd
import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

def load_csv_data(companies):
    data = []
    for company in companies:
        path = f"data/{company}.csv"
        filedata = pd.read_csv(path)
        filedata.rename(columns={"5. adjusted close": f"{company}_close", "6. volume": f"{company}_volume"}, inplace=True)
        #filedata.set_index("date", inplace=True)
        filedata = filedata[[f"{company}_close", f"{company}_volume"]]
        if len(data) == 0:
            data = filedata
        else:
            data = data.join(filedata)
        data.sort_index(inplace=True)
    return data


def calculate_returns(data, company, period):
    fut_ret = lambda current, future: 1 if future > current else 0
    data['future'] = data[f"{company}_close"].shift(-period)
    data['direction'] = list(map(fut_ret, data[f"{company}_close"], data["future"]))
    return data

def split_train_test(data, fractionForTrain):
    index = int(fractionForTrain*len(data))
    train = data[data.index < index]
    test = data[data.index >= index]
    return train, test

def preprocess_data(data, SEQ_LEN):
    data = data.drop('future', 1)
    
    for column in data.columns:
        if column != "direction":
            data[column] = data[column].pct_change()
            data.dropna(inplace=True)
            data[column] = preprocessing.scale(data[column].values)
    data.dropna(inplace=True)     
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in data.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            
    random.shuffle(sequential_data)
    '''ballance data'''
    buys = []
    sells = []
    
    for seq, direction in sequential_data:
        if direction == 0:
            sells.append([seq, direction])
        elif direction == 1:
            buys.append([seq, direction])
    
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    X = []
    Y = []
    for sequence, returnn in sequential_data:
       X.append(seq)
       Y.append(returnn)
          
    return np.array(X), Y
    
if __name__ == "__main__":
    '''downloading data'''
    companies = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'XOM', 'HD','IBM', 'INTC', 'JNJ',
                 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'VZ', 'WMT', 'WBA', 'DIS']
    companies = ['JNJ','MRK','PFE'] #pharmaceutical sector
#    '''deleted companies : DWDP 313, GS 4923, V 2691'''  
#    '''columns are open, high, low, close, adjusted close, volume, divident amount, split coefficient'''
#    data = dd.download_json_data(companies,'daily')
#    dd.save_to_csv(data)
    SEQ_LEN = 10
    FUTURE_PERIOD_PREDICT = 1
    COMPANY_TO_PREDICT = "PFE"
    EPOCHS = 20
    BATCH_SIZE = 128
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}-company-{COMPANY_TO_PREDICT}"
    
    
    '''loading data from csv and preparing future and return data'''
    data = load_csv_data(companies)
    data = calculate_returns(data, COMPANY_TO_PREDICT, FUTURE_PERIOD_PREDICT)
    '''make train set and test set'''
    train, test = split_train_test(data, 0.9)
    '''preprocessing data'''
    train_x, train_y = preprocess_data(train, SEQ_LEN)
    test_x, test_y = preprocess_data(test, SEQ_LEN)
    
    print(f"train data: {len(train_x)} test:{len(test_x)}")
    print(f"dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
    print(f"test dont buys: {test_y.count(0)}, buys {test_y.count(1)}")
    
    model = Sequential()
    model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    
    model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    
    model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))
    
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
    
    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
    
    history = model.fit(train_x, train_y,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(test_x, test_y),
                        callbacks=[tensorboard, checkpoint])
    
    ''''Score model'''
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    ''' Save model'''
    model.save("models/{}".format(NAME))
    
    
    
    
    