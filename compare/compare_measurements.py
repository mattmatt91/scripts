
import pandas as pd
import plotly.express as px
from helpers.helpers import Helpers as hp
from os.path import join
from os import getenv


def compare():
    properties = hp.read_json('properties', 'properties.json')
    path = join(getenv("DATA_PATH"), 'results',  'merged_sensors')
    onlyfiles = hp.get_all_files_in_sub(path)
    sensors = [i for i in properties['sensors']]
    for file, sensor in zip(onlyfiles, sensors):
        evaluate_sensor(file, sensor, path)


def evaluate_sensor(file: str, sensor: str, path: str):
    df = pd.read_csv(file, decimal=',', sep=';')
    df.set_index('time', inplace=True)
    plot(df, sensor, path)


def plot(df, sensor, path):
    fig = px.line(df, title=sensor)
    path_file = join(path, 'results', 'plots', 'compare')
    hp.save_html(fig, path, sensor)


if __name__ == '__main__':
    root_path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    compare(root_path)
