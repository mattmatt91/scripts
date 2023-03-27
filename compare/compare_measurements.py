
import pandas as pd
import plotly.express as px
from pathlib import Path
from os import listdir
from helpers.helpers import Helpers as hp
from os.path import isfile, join


def compare(path: str, properties: dict):
    root_path = join(path, 'results', 'merged_sensors')
    onlyfiles = [join(root_path, f)
                 for f in listdir(root_path) if isfile(join(root_path, f))]

    sensors = [i for i in properties['sensors']]
    for file, sensor in zip(onlyfiles, sensors):
        evaluate_sensor(file, sensor, path, properties)


def evaluate_sensor(file, sensor, path, properties):
    df = pd.read_csv(file, decimal=',', sep=';')
    df.set_index('time', inplace=True)
    plot(df, sensor, path)


def plot(df, sensor, path):
    fig = px.line(df)
    samples = [i.split('_')[0] for i in df.columns]
    fig.show()
    exit()
    path_file = join(path, 'results', 'plots', 'compare')

    hp.save_fig(fig, path_file, title)
    plt.close()


if __name__ == '__main__':
    root_path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    compare(root_path)
