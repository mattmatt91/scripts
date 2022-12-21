import json as js
from os.path import join
from os import scandir
from os import listdir
from os.path import isfile, join
from pathlib import Path

# read json to dict
def read_json(folder, filename):
    with open(join(folder, filename)) as json_file:
        return js.load(json_file)
    
# extract name from path
def get_name_from_info(info: dict):
    name =  f"{info['sample']}_{info['height']}_{info['number']}"
    return  name


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
        if file.find('txt')>0:
            return join(path,file)

# returns path to json with info
def get_path_info(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        if file.find('json')>0:
            return join(path,file)
        
# mkdir if not exists
def mkdir_ifnotexits(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def save_df(df, path, name):
    Path(path).mkdir(parents=True, exist_ok=True)
    path = join(path, f'{name}.txt')
    df.to_csv(path, sep=';', decimal='.', index=True)