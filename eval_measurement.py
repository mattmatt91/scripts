import pandas as pd
import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths
from plot_measurement import plot_measurement

def evaluate_measurement(data, properties, info):
    features = info
    data = clean_data(data, properties, info)
    
    for sensor in data.columns:
        result, peaks, peak_properties, results_half, results_full =  evaluate_sensor(data, sensor, properties['sensors'][sensor]['threshold'])
        features = features | result
    plot_measurement(info, data, peak_properties, results_half, results_full, peaks)
    return data, features


def clean_data(data, properties, info):
    data.drop('time [s]', axis=1, inplace=True)
    means = data[0:100].mean()  # set zero point
    data = data - means
    flag = True
    data = data.abs()  # set all positive
    cut_before_signal = properties['cut_before_signal']
    sensor_to_cut = properties['sensor_to_cut']
    threshold = properties['sensors'][sensor_to_cut]['threshold']
    cut_after_signal = properties['cut_after_signal']
    i = 0
    while flag:  # cut relevant section
        index_to_cut = data[data['Piezo1'].gt(1)].index[i]
        if data['Piezo1'].iloc[data.index.get_loc(index_to_cut)+1] > threshold:
            flag = False
        else:
            i += 1
    data = data[index_to_cut -
                cut_before_signal: index_to_cut + cut_after_signal]
    data.apply(lambda x: round(x, 2))
    data['time [s]'] = np.round(np.arange(start=0, stop=len(
        data)*(1/info['rate']), step=1/info['rate'], dtype=float), 5)
    data.reset_index(drop=True, inplace=True)
    data.set_index('time [s]', inplace=True)
    for sensor in properties['sensors']:
        if properties['sensors'][sensor]['norm']:
            pass
            # data[sensor] = floating_mean(data[sensor])
    return data


def floating_mean(data):
    N = 25
    data_flat = np.convolve(data, np.ones(N)/N, mode='same')
    data = pd.Series(
        data_flat, index=data.index[:len(data_flat)], name=data.name)
    return data


def evaluate_sensor(data, sensor, threshold):
    peaks, peak_properties, results_half, results_full, result_dict = analyse(
        data[sensor], sensor, threshold)
    return result_dict, peaks, peak_properties, results_half, results_full


def analyse(data, sensor, threshold):
    peaks, peak_properties = find_peaks(
        data, prominence=0, width=1, distance=20000, height=threshold)
    results_half = peak_widths(data, peaks, rel_height=0.5)
    results_full = peak_widths(data, peaks, rel_height=0.99)
    if len(peaks) > 0:
        val1 = int(results_full[2][0])
        val2 = int(results_full[3][0])
        df_peak = pd.DataFrame(data.iloc[val1:val2])
    else:
        df_peak=pd.DataFrame()
    # functions for feature extraction

    def get_peak():
        return data.index[int(peaks[0])]

    def get_start():
        return data.index[int(results_full[2])]

    def get_stop():
        return data.index[int(results_full[3])]

    def get_width():
        return data.index[int(results_full[3])] - data.index[int(results_full[2])]

    def get_width_half():
        return data.index[int(results_half[3])] - data.index[int(results_half[2])]

    def get_height():
        return data[data.index[int(peaks[0])]]

    def get_integral():
        return np.trapz(df_peak[sensor], x=df_peak.index)

    def get_slope():
        x = df_peak.values
        t = df_peak.index
        end = 0
        flag = False
        for i in range(len(x)-1):
            if flag == False:
                if x[i+1] > x[i]:
                    pass
                else:
                    end = i
                    flag = True
        slope = (x[end]-x[0])/(t[end]-t[0])
        return slope

    def get_width_heigth():
        return (data.index[int(results_full[3])] - data.index[int(results_full[2])])/(data[data.index[int(peaks[0])]])

    values = [get_peak, get_start, get_stop, get_width, get_width_half,
              get_height, get_integral, get_slope,  get_width_heigth]
    features = "peak[s] start[s] stop[s] width[s] width_half[s] height intetegral[Vs] slope[V/s] width_heigth[s/V]".split()

    # build the json result for this measurement
    result_dict = {}
    for feature, value in zip(features, values):
        if len(peaks)>0:
            result_dict[f"{sensor}_{feature}"] = value()
        else:
            result_dict[f"{sensor}_{feature}"] = False

    return (peaks, peak_properties, results_half, results_full, result_dict)
