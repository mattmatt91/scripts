from helpers import read_json, get_name_from_info, get_subfolders, clean_info_meaurement, get_path_data, mkdir_ifnotexits , get_path_info
import pandas as pd
from os.path import join
from eval_measurement import evaluate_measurement
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def init_data(properties):
    data = {'name': [], 'time': []}
    for sensor in properties['sensors']:
        data[sensor] = []
    return data


# reading all measurements and creating files with all measurements for each sensor
def scan_folder(path, properties):
    subfolders = get_subfolders(path)
    data = init_data(properties)
    results = []
    for folder in subfolders:
        if folder.find('\\Results') < 0 and folder.find('\\Bilder', ) < 0 and folder.find('\\results') < 0:
            info_measurement = clean_info_meaurement(
                read_json(folder, 'info.json'))
            name = get_name_from_info(info_measurement)
            path_data = get_path_data(folder)
        
            data_measurement = pd.read_csv(path_data, decimal='.', sep='\t')
            data_measurement, features = evaluate_measurement(data_measurement, properties, info_measurement)
            results.append(features)
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

def merge_results(result, folder):
    df_result = pd.DataFrame(result)
    print(df_result)
    path_result = mkdir_ifnotexits(join(folder, 'results'))
    path_to_save = join(path_result, 'results.txt')
    df_result.to_csv(path_to_save, decimal='.', sep='\t')

def merge_measurements(data, folder):
    data_sensors = {}
    for sensor in data:
        if sensor != 'name' and sensor != 'time':
            df = pd.DataFrame(data[sensor], index=data['name']).T
            df['time'] = data['time'][0]
            df.set_index('time', inplace=True)
            path_result = mkdir_ifnotexits(
                join(folder, 'results', 'merged_sensors'))
            path_to_save = join(path_result, f'{sensor}.txt')
            df.to_csv(path_to_save, decimal='.', sep='\t')
            data_sensors[sensor] = df
    return data_sensors



