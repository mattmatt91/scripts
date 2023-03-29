from helpers.helpers import Helpers as hp
from os.path import join
import pandas as pd
import os


def get_statistics(features: dict, infos: dict):
    print('processing statistics...')
    samples = infos['sample'].unique().tolist()
    statistics_list = {}
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    path = join(os.getenv("DATA_PATH"), 'results', 'statistics')
    for sample in samples:
        index = infos[infos['sample'] == sample].index.tolist()
        df_sample = features.iloc[index].describe()
        hp.save_df(df_sample, path, sample)
        statistics_list[sample] = df_sample
        df_mean[sample] = df_sample.T['mean']
        df_std[sample] = df_sample.T['std']
    hp.save_df(df_mean.T, path, 'mean')
    hp.save_df(df_std.T, path, 'std')
