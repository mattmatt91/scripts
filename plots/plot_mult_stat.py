import matplotlib.pyplot as plt
import matplotlib
from os.path import join
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px
import seaborn as sns
import warnings
import os
from os import path


def plot_components(x_r: pd.DataFrame, properties: dict, infos:dict, name=None):
    colors_dict = {}
    for i in x_r.index.unique():
        colors_dict[i] = properties['colors_samples'][i]
    fig = px.scatter_3d(
        x_r,
        x=x_r.columns[0],
        y=x_r.columns[1],
        z=x_r.columns[2],
        color_discrete_map=colors_dict,
        color=x_r.index,
        hover_data=infos.to_dict('series')
    )
    # saving plot
    path = join(os.getenv("DATA_PATH"), 'statistics')
    save_html(fig, path, name)
    # fig.show()


def plot_all_laodings(df, path, plot_properties, colors):
    fig = px.histogram(df, barmode='group',
                       x="PC",
                       y="value",
                       color='sensor',
                       color_discrete_map=colors)

    path = join(os.getenv("DATA_PATH"), 'plots', 'statistics')
    # fig.show()
    save_html(fig, path, 'all_loadings')


def plot_sum_laodings(df, path, plot_properties, colors):
    fig = px.histogram(df,
                       x="sensor",
                       y="value_abs",
                       color='sensor',
                       color_discrete_map=colors)
    # fig.show()
    save_html(fig, path, 'sum_loadings')

def plot_loadings_heat(df, properties):
    # preparing dataframe
    df = convert_df_pd(df)
    df['value_abs'] = df['value'].abs()
    df['value_abs_norm'] = normalize_data(df['value_abs'])
    df['value_norm'] = normalize_data(df['value'])

    colors = {}
    for sensor in df['sensor'].unique():
        colors[sensor] = properties['sensors'][sensor]['color']
    path = join(os.getenv("DATA_PATH"), 'plots', 'statistics')
    plot_all_laodings(df, path, properties, colors)
    plot_sum_laodings(df, path, properties, colors)


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def convert_df_pd(df):
    pcs = 'PC1 PC2 PC3'.split()
    converted = []
    for i, m, k in zip(df['sensors'], df['features'], range(len(df['features']))):
        for n in pcs:
            converted.append(
                {'sensor': i, 'feature': m, 'PC': n, 'value': df.iloc[k][n]})
    df_converted = pd.DataFrame(converted)
    return df_converted


def save_html(html_object, path, name):
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.html'
    print(path)
    html_object.write_html(path)
