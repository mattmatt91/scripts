from helpers import read_json, get_name_from_info, get_subfolders, clean_info_meaurement, get_path_data, mkdir_ifnotexits
import pandas as pd
from os.path import join


def init_data(properties):
    data = {'name': [], 'time': []}
    for sensor in properties['sensors']:
        data[sensor] = []
    return data


# reading all measurements and creating files with all measurements for each sensor
def scan_folder(path, properties):
    subfolders = get_subfolders(path)
    data = init_data(properties)

    for folder in subfolders:
        if folder.find('\\Results') < 0 and folder.find('\\Bilder', ) < 0 and folder.find('\\results') < 0:
            info_measurement = clean_info_meaurement(
                read_json(folder, 'info.json'))
            name = get_name_from_info(info_measurement)
            path_data = get_path_data(folder)
            data_measurement = pd.read_csv(path_data, decimal='.', sep='\t')
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


def merge_measurements(data, folder):
    data_sensors = {}
    for sensor in data:
        if sensor != 'name' and sensor != 'time':
            df = pd.DataFrame(data[sensor], index=data['name']).T
            df['time'] = data['time'][0]
            df.set_index('time', inplace=True)
            print(df)
            path_result = mkdir_ifnotexits(
                join(folder, 'results', 'merged_sensors'))
            path_to_save = join(path_result, f'{sensor}.txt')
            print(path_to_save)
            df.to_csv(path_to_save, decimal='.', sep='\t')
            data_sensors[sensor] = df
    return data_sensors



