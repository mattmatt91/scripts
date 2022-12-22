import json as js
from os import listdir,  scandir
from os.path import isfile, join
from pathlib import Path
from shutil import rmtree

# read json to dict


def read_json(folder, filename):
    with open(join(folder, filename)) as json_file:
        return js.load(json_file)

# extract name from path


def get_name_from_info(info: dict):
    name = f"{info['sample']}_{info['height']}_{info['number']}"
    return name


def flattern_dict(data: dict) -> dict:
    flat_dict = {}
    for key0 in data:
        if type(data[key0]) == dict:
            for key1 in data[key0]:
                if type(data[key0][key1]) == dict:
                    for key2 in data[key0][key1]:
                        if type(data[key0][key1][key2]) == dict:
                            print('dict to nested!!')
                        else:
                            flat_dict[f'{key1}_{key2}'] = data[key0][key1][key2]
                else:
                    flat_dict[f'{key0}_{key1}'] = data[key0][key1]
        else:
            flat_dict[key0] = data[key0]
    return flat_dict


# list all subfolders
def get_subfolders(path):
    return [f.path for f in scandir(path) if f.is_dir()]

# deletes obsolete infos about measurement


def clean_info_meaurement(info: dict):
    cleaned_info = {}
    for key in ['datetime', 'height', 'number', 'path', 'rate', 'sample']:
        cleaned_info[key] = info[key]
    return cleaned_info

# returns path to txt with data


def get_path_data(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        if file.find('txt') > 0:
            return join(path, file)


def del_results(path):
    folders = [f.path for f in scandir(path) if f.is_dir()]
    for folder in folders:
        if folder.find('results') > 0:
            rmtree(join(path, folder))


# get path one layer up
def one_layer_back(path: str) -> str:
    new_path = path[:path.rfind('\\')]
    return new_path

# returns path to json with info


def get_path_info(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        if file.find('json') > 0:
            return join(path, file)

# mkdir if not exists


def mkdir_ifnotexits(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_df(df, path, name, index=True):
    Path(path).mkdir(parents=True, exist_ok=True)
    path = join(path, f'{name}.txt')
    df.to_csv(path, sep=';', decimal='.', index=index)
