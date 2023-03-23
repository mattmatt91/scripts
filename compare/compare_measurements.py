

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import plotly.express as px
from os import listdir
from os.path import isfile, join


def save_fig(fig, path, name):
    fig.tight_layout()
    # print(path)
    path = join(path, 'results', 'plots', 'compare')
    Path(path).mkdir(parents=True, exist_ok=True)
    path = join(path, f'{name}.jpeg')
    # print(path)
    # plt.show()
    print(path)
    fig.savefig(path)
    plt.close(fig)


# creates plots for every sensor with all measurments
def plot(df, sensor, path, names, properties):
    plot_properties = properties['plot_properties']['compare_plots']
    print('plotting {0}-data'.format(sensor))
    x_lim_plot = properties['sensors'][sensor]['x_lim_plot']
    x_lim_plot_start = x_lim_plot[0]
    x_lim_plot_end = x_lim_plot[1]
    print(df)
    for sample in df.columns.unique():

        title = sensor + '_' + sample
        fig, ax = plt.subplots(
            figsize=plot_properties['size'], dpi=plot_properties['dpi'])

        # use this for centering around peak
        # t_max = df[sample].max()
        # x_lim_plot_start = t_max - properties['x_lim_plot'][name][0]
        # x_lim_plot_end = t_max + properties['x_lim_plot'][name][1]

        ax.plot(df.index, df[sample])
        plt.xlim(x_lim_plot_start, x_lim_plot_end)
        plt.xlabel(df.index.name, fontsize=plot_properties['label_size'])
        plt.ylabel('voltage [V]', fontsize=plot_properties['label_size'])
        plt.yticks(fontsize=plot_properties['font_size'])
        plt.xticks(fontsize=plot_properties['font_size'])
        ax.grid()
        plt.show()
        exit()
        save_fig(fig, path, title)
        plt.close()


def evaluate_sensor(file, sensor, path, properties):
    df = pd.read_csv(file, decimal='.', sep='\t')
    df.set_index('time', inplace=True)
    names = df.columns.tolist()
    plot(df, sensor, path, names, properties)


def compare(path: str, properties: dict):
    root_path = join(path, 'results', 'merged_sensors')
    onlyfiles = [join(root_path, f)
                 for f in listdir(root_path) if isfile(join(root_path, f))]#
    print(onlyfiles)
    sensors = [i for i in properties['sensors']]
    for file, sensor in zip(onlyfiles, sensors):
        evaluate_sensor(file, sensor, path, properties)


if __name__ == '__main__':
    root_path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    compare(root_path)
