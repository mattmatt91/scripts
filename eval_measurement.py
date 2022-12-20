import pandas as pd
import numpy as np
def evaluate_measurement(data, properties, info):
    features = {}
    data = clean_data(data, properties, info)
    return data, features


def clean_data(data, properties, info):
    print(info)
    means = data[0:100].drop('time [s]', axis=1, inplace=True).mean() # set zero point
    data = data - means 
    flag = True
    data = data.abs() # set all positive
    cut_before_signal = properties['cut_before_signal']
    sensor_to_cut = properties['sensor_to_cut']
    threshold = properties['sensors'][sensor_to_cut]['threshold']
    cut_after_signal = properties['cut_after_signal']
    i = 0
    while flag: #cut relevant section
        index_to_cut = data[data['Piezo1'].gt(1)].index[i] 
        if data['Piezo1'].iloc[data.index.get_loc(index_to_cut)+1] > threshold:
            flag = False
        else:
            i +=1    
    data = data[index_to_cut - cut_before_signal: index_to_cut + cut_after_signal]
    data['time'] = np.arange(0,len(data), info['rate'])
    # for sensor in properties['sensors']:
    #     if properties['sensors'][sensor]['norm']:
    #         # data[sensor] = floating_mean(data[sensor])
    return data

def floating_mean(data):
    N = 25
    data_flat = np.convolve(data, np.ones(N)/N, mode='valid')
    data = pd.Series(data_flat, index=data.index[:len(data_flat)], name= data.name)
    return data

def new_index(data):
    return np.arange()