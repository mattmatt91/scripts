from helpers.helpers import Helpers as hp
from os.path import join
import pandas as pd
import os

def get_statistics(features:dict, infos:dict):
    samples = infos['sample'].unique().tolist()
    statistics_list = {}
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    for sample in samples:
        df_sample = features[infos['sample'] == sample].describe()
        hp.save_df(df_sample, path, sample)
        statistics_list[sample] = df_sample
        df_mean[sample] = df_sample.T['mean']
        df_std[sample] = df_sample.T['std']
    path = join(os.getenv("DATA_PATH"), 'results', 'statistics')
    hp.save_df(df_mean.T, path, 'mean')
    hp.save_df(df_std.T, path, 'std')