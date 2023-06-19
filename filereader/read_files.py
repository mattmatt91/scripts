from helpers.helpers import Helpers as hp
import pandas as pd
from os.path import join
from filereader.eval_measurement import evaluate_measurement
import warnings
from os import getenv
warnings.simplefilter(action="ignore", category=RuntimeWarning)


# reading all measurements and creating files with all measurements for each sensor
def scan_folder() -> None:
    path = getenv("DATA_PATH")
    properties = hp.read_json('properties', 'properties.json')
    subfolders = [f for f in hp.get_subfolders(path) if f.find('result') < 0]
    flag_first = True  # set flag for initing csv with header
    print('reading files...')
    results = []
    for folder in subfolders:
        data_measurement, features = evaluate_measurement(properties, folder)
        results.append(features)
        append_measurement(data_measurement, path, flag_first, features)
        flag_first = False
    save_results(results, path)


def save_results(results: list, folder: str):
    print('saving results')
    df_result = pd.DataFrame(results)
    path_result = hp.mkdir_ifnotexits(join(folder, 'results'))
    path_to_save = join(path_result, 'results.csv')
    df_result.fillna(0).to_csv(path_to_save, decimal=',', sep=';', index=False)


def append_measurement(data: pd.DataFrame, folder: str, first: bool, features: dict):
    # data = data.iloc[:15]
    for sensor in data.columns:
        path_result = hp.mkdir_ifnotexits(
            join(folder, 'results', 'merged_sensors'))
        path_to_save = join(path_result, f'{sensor}.txt')
        if first:
            line = convert_to_line(data.index.to_list(), 'time')
            line += convert_to_line(data[sensor].to_list(), features['name'])
            create_text_file(path_to_save, line)
        else:
            line = convert_to_line(data[sensor].to_list(), features['name'])
            append_to_text_file(path_to_save, line)
           

def convert_to_line(data:list, name:str):
    response = f"{name}\t"
    for item in data:
        response += f"{str(item)}\t"
    response += '\n'
    return response


def create_text_file(file_path, initial_text):
    with open(file_path, 'w') as file:
        file.write(initial_text)
   

def append_to_text_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text)
 