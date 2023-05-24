import pandas as pd
from plots.plot_measurement import plot_measurement_stacked, plot_measurement
from helpers.helpers import Helpers as hp
from filereader.extract_features import extract_features
import os
from filereader.preprocessing import PreProcessing as pp
import webbrowser
import json as js
new = 1
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
    # reduce lenght of data
    data = pp.cut_time_section(data, properties)
    # add new timeaxis
    data = pp.create_time_axis(data, features)
    # eval sensor
    features = evaluate_sensors(data, properties, features)
    # plot measurements
    plot(data, properties, features)
    features = clean_before_return(features)
    return data, features


def clean_before_return(features: dict):
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
        features_sensor = extract_features(data_sensor, n_stabw)
        features['sensors'][sensor] = features_sensor
    return features


def plot(data: pd.DataFrame, properties: dict, features: dict):
    plot_measurement(data,  properties, features)
    plot_measurement_stacked(data,  properties, features)


def read_properties(path):
    print(path)
    info_raw = hp.read_json(path, 'properties.json')
    info_raw = add_combustion_param_and_ball_to_info(info_raw, path)
    info = hp.clean_info_meaurement(info_raw)
    path = os.path.join(path, 'data.csv')
    features = info
    features['name'] = f"{info['sample']}_{info['number']}"
    features['sensors'] = {}
    return path, features


def add_combustion_param_and_ball_to_info(info_raw: dict, path: str):
    flag = False
    # check data from save combustion
    # check combustion
    if path.find('safe_combustion') >= 0:
        # check combustion
        if info_raw['sample'].find('BLANK') >= 0:
            info_raw['combustion'] = "none"
            flag = True
        elif info_raw['sample'].find('Salt') >= 0:
            info_raw['combustion'] = "none"
            flag = True
        else:
            info_raw['combustion'] = "full"
            flag = True
        # check ballsize
        if not 'ball' in info_raw.keys():
            if path.find('N3') >= 0:
                info_raw['ball'] = 15
            elif path.find('15mm') >= 0:
                info_raw['ball'] = 15
            else:
                info_raw['ball'] = 10
            flag = True

    # check combustion for unknown measurements
    else:
        if not "combustion" in info_raw.keys():
            url = os.path.join(path, 'quickplot.html')
            webbrowser.open(url, new=new)
            print(f"measurement: {path}")
            combustion_input = input(
                'press \n1 for combustion, \n2 for partial combustion, \n3 for no combustion\n')
            if int(combustion_input) == 1:
                combustion = "full"
            elif int(combustion_input) == 2:
                combustion = "partial"
            elif int(combustion_input) == 3:
                combustion = "none"
            else:
                combustion = "unknown"
            info_raw['combustion'] = combustion
            flag = True
    if flag:
        with open(os.path.join(path, 'properties.json'), "w") as outfile:
            outfile.write(js.dumps(info_raw, indent=4))
    return info_raw
