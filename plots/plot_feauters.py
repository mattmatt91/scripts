"""
This module pltots  all extraceted features with mean and standard dev.

:copyright: (c) 2022 by Matthias Muhr, Hochschule-Bonn-Rhein-Sieg
:license: see LICENSE for more details.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os.path import join
from os import getenv




def transform_table(path, df_mean, df_stabw, properties):
    params = df_mean.T.index.tolist()
    # creating dataframe with means and errors
    for param in params:
        mean = df_mean[param]
        stabw = df_stabw[param]
        df_plot = pd.DataFrame({'mean': mean, 'stabw': stabw})


def plot_mean(path, df_plot, param, properties):
    plot_properties = properties['plot_properties']['compare_plots']
    # Create lists for the plot
    samples = df_plot.index.tolist()
    x_pos = np.arange(len(samples))
    mean = df_plot['mean']
    error = df_plot['stabw']

    fig, ax = plt.subplots()
    barlist = ax.bar(x_pos, mean, yerr=error, align='center', alpha=0.5,
                     ecolor='black', capsize=plot_properties['label_size'])

    ytitle = 'mean '
    ax.set_ylabel(ytitle)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(samples)
    ax.yaxis.grid(True)
    plt.xticks(rotation=45)
    plt.ylabel(param.replace('_', ' '), fontsize=plot_properties['label_size'])
    plt.yticks(fontsize=plot_properties['font_size'])
    plt.xticks(fontsize=plot_properties['font_size'])
    plt.tight_layout()
    # plt.show()
    save_fig(fig, path, param)
    plt.close()


def plot_features(properties: dict):
    path = join(getenv("DATA_PATH"), 'results', 'statistics')
    path_mean = join(path, 'results', 'statistics', 'statistics', 'mean.csv')
    path_stabw = join(path, 'results', 'statistics', 'statistics', 'std.csv')
    df_mean = pd.read_csv(path_mean, decimal='.', sep=';', index_col=0)
    df_stabw = pd.read_csv(path_stabw, decimal='.', sep=';', index_col=0)
    data = transform_table(df_mean, df_stabw, properties)
    print(data)
    # plot_mean(df_plot, param, properties)


if __name__ == '__main__':
    path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    plot_features(path)
