#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: liujie
@software: PyCharm
@file: main.py
@time: 2020/11/17 20:46
"""
'''
    整体思路:
        1.首先，写一个main.py文件进行神经网络的训练及测试过程
        2.将main.py中需要优化的参数(这里我们优化LSTM层数和全连接层数及每层神经元的个数)统一写到一个列表num中
        3.然后，遗传算法编写GA.py，用需要传入main.py文件的列表num当染色体，需要优化的参数是染色体上的基因
        
    main.py文件中，由于需要将所有优化的参数写到一个列表中，所以需要在文件中定义两个函数，
    分别是创建LSTM函数creat_lstm(inputs,units,return_sequences)
         创建全连接层函数creat_dense(inputs,units)
'''

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pylab as plt

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import optimizers, losses, metrics, models


# 定义LSTM函数
def create_lstm(inputs, units, return_sequences):
    '''
    定义LSTM函数
    :param inputs:输入，如果这一层是第一层LSTM层，则传入layers.Input()的变量名，否则传入的是上一个LSTM层
    :param units: LSTM层的神经元
    :param return_sequences: 如果不是最后一层LSTM，都需要保留所有输出以传入下一LSTM层
    :return:
    '''
    lstm = LSTM(units, return_sequences=return_sequences)(inputs)
    print('LSTM: ', lstm.shape)
    return lstm


def create_dense(inputs, units):
    '''
    定义Dense层函数
    :param inputs:输入，如果这一连接层是第一层全连接层，则需传入layers.Flatten()的变量名
    :param units: 全连接层单元数
    :return: 全连接层，BN层，dropout层
    '''
    # dense层
    dense = Dense(units, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu')(inputs)
    print('Dense:', dense.shape)
    # dropout层
    dense_dropout = Dropout(rate=0.2)(dense)

    dense_batch = BatchNormalization()(dense_dropout)
    return dense, dense_dropout, dense_batch


def load():
    '''
    数据集加载
    :return:
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # 数据集归一化
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


def classify(x_train, y_train, x_test, y_test, num):
    '''
    利用num及上面定义的层，构建模型
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param num: 需要优化的参数(LSTM和全连接层层数以及每层神经元的个数)，同时，也是遗传算法中的染色体
    :return:
    '''
    # 设置LSTM层参数
    lstm_num_layers = num[0]
    lstm_units = num[2:2 + lstm_num_layers]
    lstm_name = list(np.zeros((lstm_num_layers,)))

    # 设置LSTM_Dense层的参数
    lstm_dense_num_layers = num[1]
    lstm_dense_units = num[2 + lstm_num_layers: 2 + lstm_num_layers + lstm_dense_num_layers]
    lstm_dense_name = list(np.zeros((lstm_dense_num_layers,)))
    lstm_dense_dropout_name = list(np.zeros((lstm_dense_num_layers,)))
    lstm_dense_batch_name = list(np.zeros((lstm_dense_num_layers,)))

    inputs_lstm = Input(shape=(x_train.shape[1], x_train.shape[2]))

    for i in range(lstm_num_layers):
        if i == 0:
            inputs = inputs_lstm
        else:
            inputs = lstm_name[i - 1]
        if i == lstm_num_layers - 1:
            return_sequences = False
        else:
            return_sequences = True

        lstm_name[i] = create_lstm(inputs, lstm_units[i], return_sequences=return_sequences)

    for i in range(lstm_dense_num_layers):
        if i == 0:
            inputs = lstm_name[lstm_num_layers - 1]
        else:
            inputs = lstm_dense_name[i - 1]

        lstm_dense_name[i], lstm_dense_dropout_name[i], lstm_dense_batch_name[i] = create_dense(inputs,
                                                                                                units=lstm_dense_units[
                                                                                                    i])

    outputs_lstm = Dense(10, activation='softmax')(lstm_dense_batch_name[lstm_dense_num_layers - 1])

    # 构建模型
    LSTM_model = keras.Model(inputs=inputs_lstm, outputs=outputs_lstm)
    # 编译模型
    LSTM_model.compile(optimizer=optimizers.Adam(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    history = LSTM_model.fit(x_train, y_train,
                             batch_size=32, epochs=1, validation_split=0.1, verbose=1)
    # 验证模型
    results = LSTM_model.evaluate(x_test, y_test, verbose=0)
    return results[1]  # 返回测试集的准确率
