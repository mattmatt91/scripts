from os.path import join
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px
import os


def plot_heat(data: pd.DataFrame):
    fig = px.imshow(data,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='tealgrn'
                    )
    fig.update_layout(coloraxis_showscale=False)
    path = join(os.getenv("DATA_PATH"), 'results', 'plots', 'statistics')
    save_html(fig, path, 'Heatplot_LDA')


def plot_components(how_to_plot: dict, x_r: pd.DataFrame, properties: dict, infos: dict, name=None):
    style_dict = {}
    for key in how_to_plot.keys():
        values =[]
        mapping =[]
        data= {"values":values, "mapping":mapping} 
        style_dict[key] = data

    print(style_dict)
    exit()
    if how_to_plot["size"] == "ballsize":
        size_list = [str(i) for i in infos['ball'].tolist()]
    elif how_to_plot["size"] == "height":
        size_list = [str(i) for i in infos['height'].tolist()]
    elif how_to_plot["size"] == "sample":
        size_list = x_r.index.to_list()

    if how_to_plot["shape"] == "ballsize":
        shape_list = [str(i) for i in infos['ball'].tolist()]
    elif how_to_plot["shape"] == "height":
        shape_list = [str(i) for i in infos['height'].tolist()]
    elif how_to_plot["shape"] == "sample":
        shape_list = x_r.index.to_list()

    if how_to_plot["color"] == "ballsize":
        colors_dict = properties['colors_ballsize']
        color_list = [str(i) for i in infos['ball'].tolist()]
    elif how_to_plot["color"] == "height":
        colors_dict = properties['colors_height']
        color_list = [str(i) for i in infos['height'].tolist()]
    elif how_to_plot["color"] == "sample":
        colors_dict = properties['colors_samples']
        color_list = x_r.index.to_list()

    print(color_list)
    print(size_list)
    print(shape_list)
    print(colors_dict)
    fig = px.scatter_3d(
        x_r,
        x=x_r.columns[0],
        y=x_r.columns[1],
        z=x_r.columns[2],
        color_discrete_map=colors_dict,
        color=color_list,
        symbol=shape_list,
        size=size_list,
        hover_data=infos.to_dict('series')
    )
    # saving plot
    path = join(os.getenv("DATA_PATH"), 'results', 'plots', 'statistics')
    save_html(fig, path, name)
    fig.show()


def plot_all_laodings(df, plot_properties, colors):
    fig = px.histogram(df, barmode='group',
                       x="PC",
                       y="value",
                       color='sensor',
                       color_discrete_map=colors)

    path = join(os.getenv("DATA_PATH"), 'results', 'plots', 'statistics')
    # fig.show()
    save_html(fig, path, 'all_loadings')


def plot_sum_laodings(df, plot_properties, colors):
    fig = px.histogram(df,
                       x="sensor",
                       y="value_abs",
                       color='sensor',
                       color_discrete_map=colors)
    # fig.show()
    path = join(os.getenv("DATA_PATH"), 'results', 'plots', 'statistics')
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
    plot_all_laodings(df, properties, colors)
    plot_sum_laodings(df, properties, colors)


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
