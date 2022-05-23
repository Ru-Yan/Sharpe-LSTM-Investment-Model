__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import numpy as np
import pandas as pd


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def single_plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data,'r',label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    x = []
    for i, data in enumerate(predicted_data):
        x.append(i)
    plt.plot(x,predicted_data, 'c'+'--',label='Predicted Data')
    plt.legend()
    plt.show()
    
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data,'g'+':',label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data,'c'+'--',label='Prediction')
    plt.legend()
    plt.show()


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
 
    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    '''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	'''
    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    x_test, y_test,origin_data = data.get_test_data(
        seq_len=configs['testdata']['sequence_length'],
        normalise=configs['testdata']['normalise']
    )

    #predictions = model.predict_sequences_multiple(x_test, configs['testdata']['sequence_length'], len(y_test))
    #predictions = model.predict_sequences_multiple(x_test, configs['testdata']['sequence_length'], configs['testdata']['sequence_length'])
    #predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    predictions = model.predict_point_by_point(x_test)
    predictions = np.reshape(predictions,(-1,1))
    lens = len(predictions)
    for i in range(lens):
        predictions[i,0] = (float(predictions[i,0])+1)*float(origin_data[i,0,0])
    

    single_plot_results_multiple(predictions,origin_data[:, -1, [0]], configs['data']['sequence_length'])
    # plot_results(predictions, y_test)
    
    show_predictions = np.reshape(predictions,(-1))
    show_origin_data = np.reshape(origin_data[:, -1, [0]],(-1))
    df4 = pd.DataFrame({"predict data":show_predictions,"true data":show_origin_data})
    df4.to_csv('predict.csv')

if __name__ == '__main__':
    main()