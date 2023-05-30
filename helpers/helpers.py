import json as js
from os import listdir,  scandir
from os.path import isfile, join
from pathlib import Path
from shutil import rmtree
from matplotlib import pyplot as plt
import pandas as pd
import math


class Helpers:
    def save_html(html_object, path: str, name: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + '\\' + name + '.html'
        print(path)
        html_object.write_html(path)

    def get_all_files_in_sub(path: str):
        return [join(path, f)
                for f in listdir(path) if isfile(join(path, f))]

    def get_key_by_value(data: dict, value: int):
        for key, val in data.items():
            if val == value:
                return key

    def sample_to_numbers(samlpes: pd.Series):
        samles_unique = samlpes.unique()
        sample_dict = {}
        for sample, i in zip(samles_unique, range(len(samles_unique))):
            sample_dict[sample] = i
        numbers = [sample_dict[s] for s in samlpes]
        return sample_dict, numbers

    def read_json(folder, filename):
        # read json to dict
        with open(join(folder, filename)) as json_file:
            return js.load(json_file)

    def get_name_from_info(info: dict):
        # extract name from path
        name = f"{info['sample']}_{info['height']}_{info['number']}"
        return name

    def get_subfolders(path):
        # list all subfolders
        return [f.path for f in scandir(path) if f.is_dir()]

    def clean_info_meaurement(info: dict):
        # deletes obsolete infos about measurement
        cleaned_info = {}
        for key in ['datetime', 'height', 'number', 'rate', 'sample', 'ball', 'combustion', 'combustion_bool', 'droptime']:
            cleaned_info[key] = info[key]
        return cleaned_info

    def get_path_data(path):
        # returns path to txt with data
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if file.find('txt') > 0:
                return join(path, file)

    def del_results(path):
        folders = [f.path for f in scandir(path) if f.is_dir()]
        for folder in folders:
            if folder.find('results') > 0:
                rmtree(join(path, folder))

    def one_layer_back(path: str) -> str:
        # get path one layer up
        new_path = path[:path.rfind('\\')]
        return new_path

    def get_path_info(path):
        # returns path to json with info
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if file.find('json') > 0:
                return join(path, file)

    def mkdir_ifnotexits(path):
        # mkdir if not exists
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def save_df(df, path, name, index=True):
        Path(path).mkdir(parents=True, exist_ok=True)
        path = join(path, f'{name}.csv')
        print(path)
        df.to_csv(path, sep=';', decimal=',', index=index)

    def save_fig(fig, path, name):
        fig.tight_layout()
        print(path, name)
        path = join(path, 'results', 'plots', 'compare')
        Path(path).mkdir(parents=True, exist_ok=True)
        path = join(path, f'{name}.jpeg')
        print(path)
        fig.savefig(path)
        plt.close(fig)
        
    def calc_droptime(height:int):
        height = height/100
        g = 9.81  # Acceleration due to gravity on Earth (in m/s^2)
        time = math.sqrt(2 * height / g)
        return time
