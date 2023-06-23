
import pandas as pd
import plotly.express as px
from helpers.helpers import Helpers as hp
from os.path import join
from os import getenv
from os.path import join
import numpy as np


def compare():
    properties = hp.read_json('properties', 'properties.json')
    path = join(getenv("DATA_PATH"), 'results',  'merged_sensors')
    onlyfiles = hp.get_all_files_in_sub(path)
    sensors = [i for i in properties['sensors']]
    # for file, sensor in zip(onlyfiles, sensors):
    #     print(file, sensor)
    for file, sensor in zip(onlyfiles, sensors):
        evaluate_sensor(file, sensor, path, properties)



def evaluate_sensor(file: str, sensor: str, path: str, properties: dict):
    print(f'file read {file}')
    df = pd.read_csv(file, decimal='.', sep='\t')

    df = df.dropna(axis='columns').T
    df.columns = df.iloc[0]
    df.drop(index=df.index[0], axis=0, inplace=True)
   
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    df = shift_peaks(df)

    samples = [i.split('_')[0] for i in df.columns.to_list()]
    colors = []
    for sample in samples:
        colors.append(properties['sample'][sample])

    plot(df, sensor, path, colors)


def plot(df: pd.DataFrame, sensor: str, path: str, colors: list):
    fig = px.line(df, title=sensor, color_discrete_sequence=colors)
    fig.update_layout(
        xaxis_title='time',
        yaxis_title='Voltage [V]'
    )
    path = join(getenv("DATA_PATH"),
                "results\\plots\\compare")
    # fig.show()
    hp.save_html(fig, path, sensor)


def shift_peaks(df:pd.DataFrame, pre_time=1000, post_time=8000, time_step=1e-5):
    peak_indices = [df.index.get_loc(i) for i in df.idxmax().to_list()]
    start_index = [int(i) - pre_time for i in peak_indices ]
    end_index = [int(i)  +post_time +1 for i in peak_indices ]
    new_data= {}
    for i, col in enumerate(df.columns):
        new_data[col] = df[col].to_list()[start_index[i]:end_index[i]]
        if len(new_data[col]) < post_time+ pre_time + 1:
            new_data[col] =  df[col].to_list()[100: 100+ post_time+ pre_time + 1]
    new_data["time"]= [i/100000 for i in range(pre_time+post_time +1)]
    new_df = pd.DataFrame(new_data)
    new_df.set_index('time', inplace=True)
    return new_df


if __name__ == '__main__':
    root_path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    compare(root_path)
