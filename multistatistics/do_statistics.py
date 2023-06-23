
from os.path import join, exists
import pandas as pd
from multistatistics.lda import calc_lda
from multistatistics.pca import calc_pca
from multistatistics.statistics import get_statistics
import os
from helpers.helpers import Helpers as hp
import numpy as np


def do_statistics(seperation_key: str, how_to_plot: dict, selector: dict, statistic=True, pca=True, lda=True):
    properties = hp.read_json('properties', 'properties.json')
    # preparing result.csv for statistics
    file_path = join(
        os.getenv("DATA_PATH"), 'results', 'results.csv')
    if not exists(file_path):
        print('no results available, please run read files from main')
    else:
        features, infos = prepare_data(file_path, selector)
        if statistic:  # simple statistics
            get_statistics(features, infos)
        if pca:
            try:
                calc_pca(features, infos, properties, how_to_plot)
            except:
                print('\nnot able to do lda!!\n')
        if lda:
            try:
                calc_lda(features, infos, properties, how_to_plot, seperation_key)
            except:
                print('\nnot able to do lda!!\n')


def prepare_data(file_path: str, selector: dict):
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
                 'combustion',
                 'combustion_bool']
    infos = df[info_cols]
    features = df.drop(columns=info_cols)
    features.index = infos['name']

    if list(selector.keys())[0] != 'none':
        key_select = list(selector.keys())[0]
        val_select = selector[key_select]
        selected_data_index = infos[infos[key_select] == val_select].index
        selected_data_names = infos[infos[key_select] == val_select]['name']
        infos = infos.loc[selected_data_index]
        features = features.loc[selected_data_names]

    return features, infos


if __name__ == '__main__':
    path = 'E:\\Promotion\Daten\\29.06.21_Paper_reduziert'
    do_statistics()
