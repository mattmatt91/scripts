import matplotlib.pyplot as plt
import matplotlib
from os.path import join
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px


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


def plot_loadings_heat(df, path, properties):
    # preparing dataframe
    df = convert_df_pd(df)
    df['value_abs'] = df['value'].abs()
    df['value_abs_norm'] = normalize_data(df['value_abs'])
    df['value_norm'] = normalize_data(df['value'])
    colors = [properties['sensors'][i]['color'] for i in df['sensor'].unique()]
    plot_properties = properties['plot_properties']['loadings_plot']

    # creating plot 1: total variance of the sensors per principal component
    sns.set_style("whitegrid")
    # Sample figsize in inches
    fig, ax = plt.subplots(
        figsize=plot_properties['size'], dpi=plot_properties['dpi'])
    ax.set_ylabel('total variance of the sensors per principal component',
                  fontsize=plot_properties['font_size'])
    ax.set_xlabel('PC', fontsize=plot_properties['font_size'])
    sns.barplot(x="PC", y="value", data=df, ax=ax, hue='sensor',
                ci=None, estimator=sum, palette=colors)
    ax.tick_params(labelsize=plot_properties['label_size'])
    ax.legend(frameon=True, fontsize=plot_properties['legend_size'])
    name = 'sensor' + '_loadings'
    save_jpeg(fig, path, name)
    # plt.show()
    plt.close()

    # creating plot 2: total variance for each sensor
    fig, ax = plt.subplots(
        figsize=plot_properties['size'], dpi=plot_properties['dpi'])
    sns.barplot(x="sensor", y="value_abs", data=df, ax=ax,
                ci=None, estimator=sum, palette=colors)
    ax.set_ylabel('total variance for each sensor',
                  fontsize=plot_properties['font_size'])
    ax.set_xlabel('sensor', fontsize=plot_properties['font_size'])
    name = 'sensor' + '_loadings_simple'
    save_jpeg(fig, path, name)
    # plt.show()
    plt.close()


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def convert_df_pd(df):
    df.reset_index(drop=True, inplace=True)
    # formt den df um sodass pc keine Spalten mehr sind
    pcs = 'PC1 PC2 PC3'.split()
    df_converted = pd.DataFrame()
    for i, m, k in zip(df['sensors'], df['features'], range(len(df['features']))):
        for n in pcs:
            df_converted = df_converted.append(
                {'sensor': i, 'feature': m, 'PC': n, 'value': df.iloc[k][n]}, ignore_index=True)
    return df_converted


def save_jpeg(jpeg_object, path, name):

    path = path + '\\plots\\statistics'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    jpeg_object.savefig(path)


def save_html(html_object, path, name):

    path = path + '\\plots\\statistics'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.html'
    print(path)
    html_object.write_html(path)
