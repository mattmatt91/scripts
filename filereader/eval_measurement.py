import pandas as pd
import numpy as np
from plots.plot_measurement import plot_measurement_stacked, plot_measurement
from helpers.helpers import Helpers as hp
from filereader.extract_features import extract_features
import os
from filereader.preprocessing import PreProcessing as pp
import json as js
from matplotlib import pyplot  as plt
# this function extracts and returns features from every measurement


def evaluate_measurement(properties: dict, folder: str):
    # read properties
    path, features = read_properties(folder)
    # read file and remove time
    data = pd.read_csv(path, decimal=',', sep=';')
    data.drop(columns=['time'], inplace=True)
    # smooth and abs data wich are tagged for in properties
    data = pp.smooth_and_abs_data(data, properties)
    # removing offset and level to 0 V
    data = pp.remove_offset(data, properties)
    # add new timeaxis
    data = pp.create_time_axis(data, features)
    # eval sensor
    features = evaluate_sensors(data, properties, features)
    # plot measurements
    plot(data, properties, features)
    features = clean_before_return(features)
    return data, features


def clean_before_return(features:dict):
    sensors = features.pop('sensors')
    for sensor in sensors:
        for feature in sensors[sensor]:
            if feature[0] != '_':
                features[f'{sensor}_{feature}'] = sensors[sensor][feature]
    return features

def evaluate_sensors(data: pd.DataFrame, properties: dict, features):
    for sensor in data.columns:
        n_stabw = properties["sensors"][sensor]["n_stabw"]
        data_sensor = data[sensor]
        featrues_sensor = extract_features(data_sensor, n_stabw)
        features['sensors'][sensor] = featrues_sensor
    return features


def plot(data: pd.DataFrame, properties: dict, features: dict):
    plot_measurement(data,  properties, features)
    plot_measurement_stacked(data,  properties, features)


def read_properties(path):
    info = hp.clean_info_meaurement(
        hp.read_json(path, 'properties.json'))
    path = os.path.join(path, 'data.csv')
    features = info
    features['name'] = f"{info['sample']}_{info['number']}"
    features['sensors'] = {}
    return path, features



