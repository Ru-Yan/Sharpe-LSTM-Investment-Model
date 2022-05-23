from keras.models import load_model
import math
import numpy as np
import pandas as pd
import json
from numpy import newaxis

#predict_dat must be bigger than squence length!
predict_day = 15

model = load_model('saved_models/bitcoin.h5') 
configs = json.load(open('config.json', 'r'))

def normalise_windows(window_data, single_window=False):
    '''Normalise window with a base value of zero'''
    normalised_data = []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window = []
        for col_i in range(window.shape[1]):
            normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
            normalised_window.append(normalised_col)
        normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
        normalised_data.append(normalised_window)
    return np.array(normalised_data)

def predict_sequences_multiple(data, window_size,origin_data):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    print('[Model] Predicting Sequences Multiple...')
    origin_window = np.array(origin_data, copy=True)
    i = 0
    prediction_seqs = []
    curr_frame = data
    for j in range(10):
        predicted = model.predict(curr_frame[newaxis,:,:])[0,0]
        answer = (predicted+1)*origin_window[i]
        i = i+1
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted, axis=0)
        prediction_seqs.append(answer)
    return prediction_seqs

dataframe = pd.read_csv('Output.csv')
data_all = dataframe.get(configs['data']['columns']).values[:predict_day+1]
len_all  = len(data_all)
len_train_windows = None
data_windows = []
for i in range(len_all - configs['testdata']['sequence_length']):
    data_windows.append(data_all[i:i+configs['testdata']['sequence_length']])

data_windows = np.array(data_windows).astype(float)
origin_data = np.array(data_windows, copy=True)
origin_data = origin_data[-1]
data_windows = normalise_windows(data_windows, single_window=False)

x = data_windows[:, :-1]
y = data_windows[:, -1, [0]]
x = x[-1,...]

predictions = predict_sequences_multiple(x, 10,origin_data)
predictions = np.reshape(predictions,(-1,1))
avg = np.sum(predictions) / len(predictions)
print("origin: ",origin_data[-1],' ',"pre_avg: ",avg,' ',"increase or decrease: ",float((avg-origin_data[-1])/origin_data[-1])*100,'%')





