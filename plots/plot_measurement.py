import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import plotly.express as px
from helpers.helpers import Helpers as hp
from os.path import join
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"


def plot_measurement(df: pd.DataFrame, properties: dict, features: dict):
    fig = df.plot()
    for item in fig.data:
        color = properties['sensors'][item.name]['color']
        item.update(line_color=color)
    path = join(os.getenv("DATA_PATH"),
                "results\\plots\\measurements\\unstacked")
    hp.mkdir_ifnotexits(path)
    name = features['name']
    path = join(path, f'{name}.html')
    print(path)
    fig.write_html(path)


def plot_measurement_stacked(df: pd.DataFrame, properties: dict, features: dict):
    titles = tuple(df.columns)
    n_rows = len(df.columns)

    fig = make_subplots(rows=n_rows,
                        cols=1,
                        row_titles=titles,
                        x_title=df.index.name)

    traces = []
    rows = []
    for i, sensor in zip(range(len(df.columns)), df.columns):
        traces, rows = draw_sensor(
            traces, rows, df, features, properties, sensor, i)

    cols = [1 for _ in rows]
    fig.add_traces(traces, cols=cols, rows=rows)
    name = features['name']

    fig.update_layout(height=1800,
                      width=1000,
                      title_text=name,
                      showlegend=False)

    path = join(os.getenv("DATA_PATH"),
                "results\\plots\\measurements\\stacked")
    hp.mkdir_ifnotexits(path)
    path = join(path, f'{name}.html')
    print(path)
    # fig.show()
    fig.write_html(path)


def draw_sensor(traces, rows, df, features, properties, sensor, row):
    color = properties['sensors'][sensor]['color']
    trace = go.Scatter(
        x=df.index,
        y=df[sensor],
        line=dict(color=color)
    )
    rows.append(row+1)
    traces.append(trace)

    if len(features['sensors'][sensor]) > 0:
        traces_peak = draw_peak(df, features, sensor, color)
        rows = rows + [row+1 for _ in traces_peak]
        traces = traces + traces_peak
    return traces, rows


def draw_peak(df, features, sensor, color):
    traces = []
    traces.append(draw_vert(features, sensor, 'green'))
    traces.append(draw_base(features, sensor, 'orange'))
    traces.append(draw_half(features, sensor, 'blue'))
    return traces


def draw_vert(features: dict, sensor: str, color: str):
    features_plot = features['sensors'][sensor]
    x0 = features_plot['_peak_x']
    x1 = features_plot['_peak_x']
    y0 = features_plot['_threshold']
    y1 = features_plot['peak_y']
    return draw_line(x0, x1, y0, y1, color=color)


def draw_base(features: dict, sensor: str, color: str):
    features_plot = features['sensors'][sensor]
    x0 = features_plot['_base_x1']
    x1 = features_plot['_base_x2']
    y0 = features_plot['base_y1']
    y1 = features_plot['base_y2']
    return draw_line(x0, x1, y0, y1, color=color)


def draw_half(features: dict, sensor: str, color: str):
    features_plot = features['sensors'][sensor]
    x0 = features_plot['_half_x1']
    x1 = features_plot['_half_x2']
    y0 = features_plot['half_y1']
    y1 = features_plot['half_y2']
    return draw_line(x0, x1, y0, y1, color=color)


def draw_line(x0, x1, y0, y1, color):
    trace = go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        line=dict(color=color,
                  width=4,
                  dash="dot")
    )
    return trace
