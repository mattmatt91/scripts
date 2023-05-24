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


def create_mapping_size(data: list):
    mapping = {}
    n = 10
    for i in set(data):
        mapping[i] = n
        n += 10
    numeric_values = [mapping[i] for i in data]
    return numeric_values


def plot_components(how_to_plot: dict, x_r: pd.DataFrame, properties: dict, infos: dict, name=None):
    style_dict = {}
    for key in how_to_plot.keys():
        if key != 'none':
            if key == 'color':
                mapping = properties[how_to_plot[key]]
                values = [str(i) for i in infos[how_to_plot[key]].tolist()]
            elif key == 'size':
                mapping = []
                values = infos[how_to_plot[key]].tolist()
                if type(values[0]) == str:
                    values = create_mapping_size(values)
            elif key == 'symbol':
                mapping = []
                values = infos[how_to_plot[key]].tolist()

            data = {"values": values, "mapping": mapping}
            style_dict[key] = data


    fig = px.scatter_3d(
        x_r,
        x=x_r.columns[0],
        y=x_r.columns[1],
        z=x_r.columns[2],
        color_discrete_map=style_dict['color']['mapping'],
        color=style_dict['color']['values'],
        symbol=style_dict['symbol']['values'],
        size=style_dict['size']['values'],
        hover_data=infos.to_dict('series')

    )
    # saving plot
    legend_color = how_to_plot['color']
    legend_symbol = how_to_plot['symbol']
    legend_heaer = f'{legend_color}\t{legend_symbol}'
    fig.update_layout(legend_title_text=legend_heaer)
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
