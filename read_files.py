from os import listdir, scandir, sep
from os.path import isfile, join
from xml.etree.ElementPath import prepare_parent
from sensors import read_file
from sensors import Sensor

from read_spectra import read_spectra, Spectra

import pandas as pd
from pathlib import Path
import json as js


def read_json(filename):
    with open(filename) as json_file:
        return js.load(json_file)

def extract_info(path):
    mydict={}
    path = path + '\\info.json'
    mydict = read_json(path)
    return mydict

def extract_properties(): # not used in this file 
    path = str(Path().absolute()) + '\\properties.json'
    print(path)
    dict = read_json(path)
    return dict

def scan_folder(path, properties):
    subfolders = [f.path for f in scandir(path) if f.is_dir()]
    #cerating dataframes and objects to save results
    df_result = pd.DataFrame()
    df_result_raw = Sensor(properties) # dataframe for each sensor with all measurements


    # creates list with subfolers
    for folder in subfolders:
        if folder.find('\\Results') < 0 and folder.find('\\Bilder', ) < 0 and folder.find('\\results') < 0:
            mydict = extract_info(folder)
            name = mydict['path'][mydict['path'].rfind('\\')+1:]
            mydict.update({"name": name})
            mydict.update(read_file(folder, '.', name, path, df_result_raw, properties)) #evaluating file
            print(mydict)
            exit()

            # nur für alte daten mit spektrometer
            mydict.update(read_spectra(folder, properties, name))
            
            df_result = df_result.append(mydict, ignore_index=True) # append measurement in result file
    result_path = path + '\\' + 'Results'
    Path(result_path).mkdir(parents=True, exist_ok=True)
    result_path = result_path + '\\Result.csv'
    df_result.to_csv(result_path, decimal=',', sep=';', index = False) # safe the result df
    # df_result_raw.save_items(path) # save the sensor df
    print(path)



if __name__ == '__main__':
    scan_folder("C:\\Users\\Matthias\\Desktop\\Messaufbau\\dataaquisition\\data\\test")