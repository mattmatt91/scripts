import matplotlib.pyplot as plt
import matplotlib
from os.path import join
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px
import seaborn as sns
import warnings


def plot_components(x_r: pd.DataFrame, path: str, properties: dict, names: pd.Series, name=None):
    plot_properties = properties['plot_properties']["components_plot_html"]

    colors_dict = {}
    for i in x_r.index.unique():
        colors_dict[i] = properties['colors_samples'][i]
    fig = px.scatter_3d(
        x_r,
        x='PC1',
        y='PC2',
        z='PC3',
        color_discrete_map=colors_dict,
        color=x_r.index,
        hover_data={'name': names.tolist()}
    )

    # setting plot parameters
    fig.update_layout(
        legend_title_font_size=plot_properties['legend_size'],
        legend_font_size=plot_properties['legend_size']/1.2,
        font_size=plot_properties['font_size']
    )

    # saving plot
    save_html(fig, path, name)
    # fig.show()


def plot_all_laodings(df, path, plot_properties, colors):
    fig = px.histogram(df, barmode='group',
            x="PC",
            y="value",
            color='sensor',
            color_discrete_map=colors)
    fig.update_layout(width=plot_properties["width"], height=plot_properties["height"], bargap=0.05)
    # fig.show()
    save_html(fig, path, 'all_loadings')

def plot_sum_laodings(df, path, plot_properties, colors):
    fig = px.histogram(df,
            x="sensor",
            y="value_abs",
            color='sensor',
            color_discrete_map=colors)
    fig.update_layout(width=plot_properties["width"], height=plot_properties["height"], bargap=0.05)
    # fig.show()
    save_html(fig, path, 'sum_loadings')



def plot_loadings_heat(df, path, properties):
    # preparing dataframe
    df = convert_df_pd(df)
    df['value_abs'] = df['value'].abs()
    df['value_abs_norm'] = normalize_data(df['value_abs'])
    df['value_norm'] = normalize_data(df['value'])

    colors = {}
    for sensor in df['sensor'].unique():
        colors[sensor] = properties['sensors'][sensor]['color']
    plot_properties = properties['plot_properties']['loadings']

    plot_all_laodings(df, path, plot_properties, colors)
    plot_sum_laodings(df, path, plot_properties, colors)


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
