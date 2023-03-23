import pandas as pd
import numpy as np
from plots.plot_measurement import  plot_measurement_stacked ,plot_measurement
from hepers.helpers import clean_info_meaurement, read_json,  get_name_from_info, get_path_data, del_results, one_layer_back
from filereader.extract_features import extract_features, get_peak
import os


def evaluate_measurement(properties: dict, folder: str):
    info = clean_info_meaurement(
        read_json(folder, 'info.json'))
    name = get_name_from_info(info)
    path = get_path_data(folder)
    # del_results(folder)

    features = info
    features['name'] = name
    features['sensors'] = {}
    data = pd.read_csv(path, decimal='.', sep='\t')
    data = clean_data(data, properties, info)
    for sensor in data.columns:
        featrues_sensor = evaluate_sensor(
            data[sensor], sensor, properties['sensors'][sensor]['threshold'])
        features['sensors'][sensor] = featrues_sensor

    plot_measurement(data, features, properties, name, one_layer_back(folder))
    plot_measurement_stacked(data, features, properties,
                             name, one_layer_back(folder))
    return data, features, name


def cut_time_section(data: pd.DataFrame, properties: dict) -> pd.DataFrame:
    sensor_to_cut = properties['sensor_to_cut']
    threshold = properties['sensors'][sensor_to_cut]['threshold']
    cut_before_signal = properties['cut_before_signal']
    cut_after_signal = properties['cut_after_signal']
    flag = True
    i = 0
    while flag:  # cut relevant section
        index_to_cut = data[data['Piezo1'].gt(1)].index[i]
        if data['Piezo1'].iloc[data.index.get_loc(index_to_cut)+1] > threshold:
            flag = False
        else:
            i += 1
    data = data[index_to_cut -
                cut_before_signal: index_to_cut + cut_after_signal]
    return data


def clean_data(data, properties, info):
    # cleaning
    data.drop('time [s]', axis=1, inplace=True)
    means = data[0:properties['points_offset']].mean()  # set zero point
    data = data - means
    data = data.abs()  # set all positive
    data = cut_time_section(data, properties)
    data.apply(lambda x: round(x, 2))
    # adding new time axis
    data['time [s]'] = np.round(np.arange(start=0, stop=len(
        data)*(1/info['rate']), step=1/info['rate'], dtype=float), 5)
    data.reset_index(drop=True, inplace=True)
    data.set_index('time [s]', inplace=True)
    # smooth data
    for sensor in properties['sensors']:
        if properties['sensors'][sensor]['norm'] > 1:
            data[sensor] = floating_mean(
                data[sensor], n=properties['sensors'][sensor]['norm'])
    return data


def floating_mean(data, n=25):
    data_flat = np.convolve(data, np.ones(n)/n, mode='same')
    data = pd.Series(
        data_flat, index=data.index[:len(data_flat)], name=data.name)
    return data


def evaluate_sensor(data: pd.Series, sensor: str, threshold: float):
    peak_info = get_peak(data, threshold)
    # build the json result for this measurement
    if len(peak_info) > 0:
        feautures = extract_features(data, peak_info)
        return feautures
    else:
        return {}
