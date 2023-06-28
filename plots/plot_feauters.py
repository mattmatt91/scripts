import pandas as pd
from os.path import join
from os import getenv
from helpers.helpers import Helpers as hp
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

info_cols = ['datetime',
             'height',
             'number',
             'sample',
             'name',
             'ball',
             'rate',
             'combustion',
             'combustion_bool']


def plot_features():
    path = join(getenv("DATA_PATH"), 'results', 'results.csv')
    data = pd.read_csv(path, sep=';', decimal=',')
    # info = data[info_cols]
    # data.drop(info_cols, inplace=True)

    new_data = []
    for feature in data:
        if feature not in info_cols:
            for height in set(data['height']):
                for sample in set(data['sample']):
                    this_data = {}
                    this_data = data[(data['sample'] == sample)
                                     & (data['height'] == height)
                                     & (data['combustion_bool'] == True)]
                    mean = this_data[feature].mean()
                    std = this_data[feature].std()
                    this_data = {'mean': mean, 'feature': feature,
                                 'height': height, 'sample': sample, 'std': std}
                    new_data.append(this_data)
    df = pd.DataFrame(new_data)

    plot_all_that_stuff(df)


def plot_all_that_stuff(df: pd.DataFrame):
    for feature in set(df['feature']):
        sample_colors = hp.read_json('properties', 'properties.json')['sample']
        # colors = [sample_colors[i] for i in [str(n) for n in set(df[sep])]]
        fig = px.bar(df[df['feature'] == feature], x="height", y="mean",
                     color="sample", barmode="group", error_y='std', title=feature)
        fig.update_xaxes(type='category')
        fig.update_xaxes(categoryorder='category ascending')
        path = join(getenv("DATA_PATH"), 'results',
                    'plots',  'statistics', 'means')
        hp.save_html(fig, path, feature)
        # fig.show()
        # exit()


if __name__ == '__main__':
    path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    plot_features(path)
