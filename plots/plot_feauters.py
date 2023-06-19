import pandas as pd
from os.path import join
from os import getenv
from helpers.helpers import Helpers as hp
import plotly.express as px


def plot_statistics(df_mean: pd.DataFrame, df_stabw: pd.DataFrame):
    params = df_mean.T.index.tolist()
    # creating dataframe with means and errors
    for param in params:
        mean = df_mean[param]
        stabw = df_stabw[param]
        df_plot = pd.DataFrame({'mean': mean, 'stabw': stabw})
        plot_mean(df_plot, param)


def plot_mean(data: pd.DataFrame, param: str):
    path = join(getenv("DATA_PATH"), 'results', 'plots',  'statistics', 'means')
    print()
    sample_colors = hp.read_json('properties', 'properties.json')["sample"]
    colors = [sample_colors[i] for i in data.index]
    fig = px.bar(data, x=data.index, y="mean",
                 error_y="stabw",
                 title=param)
    fig.update_traces(marker_color=colors)
    hp.save_html(fig, path, param)


def plot_features():
    path = join(getenv("DATA_PATH"), 'results', 'statistics')
    df_mean = pd.read_csv(join(path, 'mean.csv'),
                          decimal=',', sep=';', index_col=0)
    df_stabw = pd.read_csv(join(path, 'std.csv'),
                           decimal=',', sep=';', index_col=0)

    plot_statistics(df_mean, df_stabw)


if __name__ == '__main__':
    path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    plot_features(path)
