import pandas as pd
from os.path import join
from os import getenv
from helpers.helpers import Helpers as hp
import plotly.express as px
import numpy as np

info_cols = ['datetime',
             'height',
             'number',
             'sample',
             'name',
             'ball',
             'rate',
             'combustion',
             'combustion_bool']


def plot_features(sep: str):
    path = join(getenv("DATA_PATH"), 'results', 'results.csv')
    data = pd.read_csv(path, sep=';', decimal=',')
    classes = list(set(data[sep]))
    new_data = []
    for myclass in classes:
        class_data = data[data[sep] == myclass]
        class_data.drop(info_cols, axis=1, inplace=True)
        for feature in class_data.columns:
            mean = np.abs(class_data[feature].mean())
            stabw = class_data[feature].std()
            new_data.append({'mean': mean,
                             'stabw': stabw,
                             'feature': feature,
                             sep: myclass})

    df = pd.DataFrame(new_data)
    plot_all_that_stuff(df, sep)


def plot_all_that_stuff(df: pd.DataFrame, sep: str):
    for feature in set(df['feature']):
        sample_colors = hp.read_json('properties', 'properties.json')[sep]
        colors = [sample_colors[i] for i in [str(n) for n in set(df[sep])]]
        fig = px.bar(df[df["feature"] == feature], x=sep, y="mean",
                     error_y="stabw",
                     title=feature)
        fig.update_traces(marker_color=colors)
        path = join(getenv("DATA_PATH"), 'results',
                    'plots',  'statistics', 'means')
        hp.save_html(fig, path, feature)
        # exit()


if __name__ == '__main__':
    path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    plot_features(path)
