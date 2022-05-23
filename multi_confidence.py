from keras.models import load_model
import math
import numpy as np
import pandas as pd
import json
from numpy import newaxis

#predict_dat must be bigger than squence length!


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
    origin_window = np.array(origin_data, copy=True)
    i = 0
    prediction_seqs = []
    curr_frame = data
    for j in range(3):#往后预测几天
        predicted = model.predict(curr_frame[newaxis,:,:])[0,0]
        answer = (predicted+1)*origin_window[i]
        i = i+1
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted, axis=0)
        prediction_seqs.append(answer)
    return prediction_seqs

sum = 0
sum_true = 0
error_more = 0
error_less = 0

def confidence(predict_day):
    global sum
    global sum_true
    global error_more
    global error_less
    window_len = 3#窗口大小
    dataframe = pd.read_csv('Output.csv')
    data_all = dataframe.get(configs['data']['columns']).values[:predict_day+1]
    len_all  = len(data_all)
    len_train_windows = None
    data_windows = []
    for i in range(len_all - window_len):
        data_windows.append(data_all[i:i+window_len])

    data_windows = np.array(data_windows).astype(float)
    origin_data = np.array(data_windows, copy=True)
    origin_data = origin_data[-1]
    data_windows = normalise_windows(data_windows, single_window=False)

    x = data_windows[:, :-1]
    y = data_windows[:, -1, [0]]
    x = x[-1,...]

    predictions = predict_sequences_multiple(x, window_len,origin_data)
    predictions = np.reshape(predictions,(-1,1))
    avg = np.sum(predictions) / len(predictions)
    ori = origin_data[-1]
    avg = 0.0983*avg + 0.9017*ori
    pre_iod = float((avg-origin_data[-1])/origin_data[-1])
    true_iod = float(dataframe.get(configs['data']['columns']).values[predict_day] - origin_data[-1]) / origin_data
    true_iod = true_iod[0]
    if(pre_iod*true_iod >= 0):
        sum_true = sum_true + 1
    elif(pre_iod >= 0):
        error_more = error_more + 1
    else:
        error_less = error_less + 1
    sum = sum + 1
    print('pre_data: ',avg,' ','origin_data: ',ori,' ',"increase or decrease: ",' ',pre_iod,'%',' ','true_iod: ',true_iod,' ','divide: ',pre_iod-true_iod,' ','times: ',pre_iod/true_iod,' ','sum: ',sum,' ','sum_true: ',sum_true,"true_percent: ",sum_true / sum,' ','error_more: ',error_more,' ','error_less: ',error_less)
    return origin_data[-1],avg,float((avg-origin_data[-1])/origin_data[-1])*100

if __name__ == '__main__':
    dataframe = pd.read_csv('Output.csv')
    window_len = 3#窗口大小
    true = []
    con = []
    for i in range(window_len,1826):
        a,b,c = confidence(i)
        true.append(a)
        con.append(c)
    show_true = np.reshape(true,(-1))
    show_con_data = np.reshape(con,(-1))
    df4 = pd.DataFrame({"confidence":show_con_data,"true data":show_true})
    df4.to_csv('confidence.csv')


