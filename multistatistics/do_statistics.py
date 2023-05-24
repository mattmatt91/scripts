
from os.path import join, exists
import pandas as pd
from multistatistics.lda import calc_lda
from multistatistics.pca import calc_pca
from multistatistics.statistics import get_statistics
import os
from helpers.helpers import Helpers as hp
import numpy as np


def do_statistics(seperation_key:str,how_to_plot:dict, statistic=True, pca=True, lda=True, ):
    properties = hp.read_json('properties', 'properties.json')
    # preparing result.csv for statistics
    file_path = join(
        os.getenv("DATA_PATH"), 'results', 'results.csv')
    if not exists(file_path):
        print('no results available, please run read files from main')
    else:
        features, infos = prepare_data(file_path)
        if statistic:  # simple statistics
            get_statistics(features, infos)
        if pca:
            calc_pca(features, infos, properties, how_to_plot)
        if lda:
            calc_lda(features, infos, properties, how_to_plot, seperation_key)


def prepare_data(file_path):
    print(file_path)
    df = pd.read_csv(file_path, delimiter=';', decimal=',')
    df.fillna(0)  # features without values filled with 0.
    info_cols = ['datetime',
                 'height',
                 'number',
                 'sample',
                 'name',
                 'ball',
                 'rate',
                 'combustion']
    infos = df[info_cols]
    features = df.drop(columns=info_cols)
    features.index = infos['name']
    # features = features.astype(np.float128)
    return features, infos


if __name__ == '__main__':
    path = 'E:\\Promotion\Daten\\29.06.21_Paper_reduziert'
    do_statistics()
