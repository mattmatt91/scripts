from machine_learning.deep_learning import deep_learing
import pandas as pd
from os.path import join
from os import getenv

def do_machine_learning(deeplearning=True):
    file_path = join(
        getenv("DATA_PATH"), 'results', 'results.csv')
    features, infos = prepare_data(file_path)
    if deep_learing:
        deep_learing(features, infos)
        
        
def prepare_data(file_path):
    df = pd.read_csv(file_path, delimiter=';', decimal=',')
    df.fillna(0)  # features without values filled with 0.
    info_cols = ['datetime',
                 'height',
                 'number',
                 'rate',
                 'sample',
                 'name']
    infos = df[info_cols]
    features = df.drop(columns=info_cols)
    features.index = infos['name']
    return features, infos
        