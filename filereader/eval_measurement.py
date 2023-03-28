import pandas as pd
import numpy as np
from plots.plot_measurement import plot_measurement_stacked, plot_measurement
from helpers.helpers import Helpers as hp
from filereader.extract_features import extract_features, get_peak
import os
import json as js

# this function extracts and returns features from every measurement


def evaluate_measurement(properties: dict, folder: str):
    name, path, features, info = read_properties(folder)
    data = pd.read_csv(path, decimal=',', sep=';')
    data = clean_data(data, properties, info)

    # eval sensor
    for sensor in data.columns:
        threshold = properties['sensors'][sensor]['threshold']
        data_sensor = data[sensor]

        featrues_sensor = evaluate_sensor(
            data_sensor, sensor, threshold)

        features['sensors'][sensor] = featrues_sensor

    # plot every measurement
    plot_measurement(data, features, properties,
                     name, hp.one_layer_back(folder))
    plot_measurement_stacked(data, features, properties,
                             name, hp.one_layer_back(folder))
    return data, features, name


def read_properties(path):
    info = hp.clean_info_meaurement(
        hp.read_json(path, 'properties.json'))
    name = hp.get_name_from_info(info)
    path = os.path.join(path, 'data.csv')
    features = info
    features['name'] = f"{info['sample']}_{info['number']}"
    features['sensors'] = {}
    return name, path, features, info


def cut_time_section(data: pd.DataFrame, properties: dict) -> pd.DataFrame:
    sensor_to_cut = properties['sensor_to_cut']
    threshold = properties['sensors'][sensor_to_cut]['threshold']
    cut_before_signal = properties['cut_before_signal']
    cut_after_signal = properties['cut_after_signal']
    flag = True
    i = 0
    while flag:  # cut relevant section
        index_to_cut = data[data['Piezo'].gt(threshold)].index[i]
        if data['Piezo'].iloc[data.index.get_loc(index_to_cut)+1] > threshold:
            flag = False
        else:
            i += 1
    data = data[index_to_cut -
                cut_before_signal: index_to_cut + cut_after_signal]
    return data


def clean_data(data, properties, info):
    # cleaning
    time = data['time']
    data.drop('time', axis=1, inplace=True)

    # handle offset
    means = data[0:properties['points_offset']].mean()  # set zero point
    data = data - means
    for sensor in data:
        if properties['sensors'][sensor]['abs']:
            # set all positive ### go on here
            data[sensor] = data[sensor].abs()
    data = cut_time_section(data, properties)
    data.apply(lambda x: round(x, 2))

    # adding new time axis
    data = create_time_axis(data, info)

    # smooth data
    data = smooth_data(data, properties)

    return data


def create_time_axis(data: pd.DataFrame, info: dict) -> pd.DataFrame:
    data['time'] = np.round(np.arange(start=0, stop=len(
        data)*(1/info['rate']), step=1/info['rate'], dtype=float), 5)
    data.reset_index(drop=True, inplace=True)
    data.set_index('time', inplace=True)
    return data


def smooth_data(data: pd.DataFrame, properties: dict) -> pd.DataFrame:
    for sensor in properties['sensors']:
        if properties['sensors'][sensor]['smooth'] >0:
            data[sensor] = floating_mean(
                data[sensor], n=properties['sensors'][sensor]['smooth'])
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
