from helpers.helpers import  Helpers as hp
import pandas as pd
from os.path import join
from filereader.eval_measurement import evaluate_measurement
import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def init_data(properties: dict) -> dict:
    data = {'name': [], 'time': []}
    for sensor in properties['sensors']:
        data[sensor] = []
    return data


# reading all measurements and creating files with all measurements for each sensor
def scan_folder(path: str, properties: dict) -> None:
    subfolders = [f for f in hp.get_subfolders(path) if f.find('result') < 0]
    data = init_data(properties)
    results = []
    print('reading files...')
    for folder in subfolders:
        print(folder)
        data_measurement, features, name = evaluate_measurement( # features were extracted from each measurement
            properties, folder)
        results.append(hp.flattern_dict(features))
        for sensor in data:
            if sensor == 'name':
                data[sensor].append(name)
            elif sensor == 'time':
                data[sensor].append(data_measurement.index)
            else:
                if sensor in data_measurement.columns:
                    data[sensor].append(data_measurement[sensor])
                else:
                    print(f'{sensor} not in measurement {name}')

    merge_measurements(data, path)
    merge_results(results, path)


def merge_results(result: list, folder: str):
    df_result = pd.DataFrame(result)
    path_result = hp.mkdir_ifnotexits(join(folder, 'results'))
    path_to_save = join(path_result, 'results.csv')
    df_result.fillna(0).to_csv(path_to_save, decimal=',', sep=';', index=False)


def merge_measurements(data: pd.DataFrame, folder: str):
    data_sensors = {}
    for sensor in data:
        if sensor != 'name' and sensor != 'time':
            df = pd.DataFrame(data[sensor], index=data['name']).T
            df['time'] = data['time'][0]
            df.set_index('time', inplace=True)
            path_result = hp.mkdir_ifnotexits(
                join(folder, 'results', 'merged_sensors'))
            path_to_save = join(path_result, f'{sensor}.txt')
            df.to_csv(path_to_save, decimal=',', sep=';')
            data_sensors[sensor] = df
