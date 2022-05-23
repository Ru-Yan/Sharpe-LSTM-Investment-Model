from keras.models import load_model
import math
import numpy as np
import pandas as pd
import json

#predict_dat must be bigger than squence length!
predict_day = 25

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

def predict_point_by_point(data):
	#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
	print('[Model] Predicting Point-by-Point...')
	predicted = model.predict(data)
	predicted = np.reshape(predicted, (predicted.size,))
	return predicted

dataframe = pd.read_csv('Output.csv')
data_all = dataframe.get(configs['data']['columns']).values[:predict_day+1]
len_all  = len(data_all)
len_train_windows = None
data_windows = []
for i in range(len_all - configs['testdata']['sequence_length']):
    data_windows.append(data_all[i:i+configs['testdata']['sequence_length']])

data_windows = np.array(data_windows).astype(float)
origin_data = data_windows
data_windows = normalise_windows(data_windows, single_window=False)

x = data_windows[:, :-1]
y = data_windows[:, -1, [0]]

predictions = predict_point_by_point(x)
predictions = np.reshape(predictions,(-1,1))
anti_predictions = predictions
origin_predictions = np.array(predictions, copy=True)
lens = len(predictions)
for i in range(lens):
    anti_predictions[i,0] = (float(predictions[i,0])+1)*float(origin_data[i,0,0])
print(origin_predictions[predict_day-configs['testdata']['sequence_length']],' ',anti_predictions[predict_day-configs['testdata']['sequence_length']] ,' ' ,origin_data[predict_day-configs['testdata']['sequence_length'], -1, [0]])