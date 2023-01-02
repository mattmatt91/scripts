import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import plotly.express as px
from helpers import mkdir_ifnotexits
from os.path import join
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"


def plot_measurement(df, features, properties, name, path):
    fig  = df.plot(title=name)
    for item in fig.data:
        color = properties['sensors'][item.name]['color']
        item.update(line_color=color)
        

    path = f"{path}\\results\\plots\\measurements\\stacked"
    mkdir_ifnotexits(path)
    path = join(path, f'{name}.html')
    fig.write_html(path)

def plot_measurement_stacked(df, features, properties, name, path):
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
            traces, rows, df, features, properties, sensor, i, plt_peak=True)
    cols = [1 for _ in rows]
    fig.add_traces(traces, cols=cols, rows=rows)

    fig.update_layout(height=1800,
                      width=1000,
                      title_text=name,
                      showlegend=False)

    path = f"{path}\\results\\plots\\measurements\\stacked"
    mkdir_ifnotexits(path)
    path = join(path, f'{name}.html')
    fig.write_html(path)



def draw_sensor(traces, rows, df, features, properties, sensor, row, plt_peak=True):
    color = properties['sensors'][sensor]['color']
    trace = go.Scatter(
        x=df.index,
        y=df[sensor],
        line=dict(color=color)
    )
    rows.append(row+1)
    traces.append(trace)

    if len(features['sensors'][sensor]) > 0 and plt_peak:
        traces_peak = draw_peak(df, features, sensor, color)
        rows = rows + [row+1 for _ in traces_peak]
        traces = traces + traces_peak
    return traces, rows


def draw_peak(df, features, sensor, color):
    traces = []
    # traces.append(draw_base(df, features, sensor, 'orange'))
    traces.append(draw_vert(df, features, sensor, 'green'))
    traces.append(draw_full(df, features, sensor, 'orange'))
    traces.append(draw_half(df, features, sensor, 'orange'))
    return traces


def draw_base(df, features, sensor, color):
    x0 = df.index[int(features['sensors'][sensor]['left_ips'])]
    x1 = df.index[int(features['sensors'][sensor]['right_ips'])]
    y0 = df[sensor].loc[x0]
    y1 = df[sensor].loc[x1]
    return draw_line(x0, x1, y0, y1, color=color)


def draw_vert(df, features, sensor, color):
    x0 = features['sensors'][sensor]['peak_time']
    x1 = x0
    y0 = 0
    y1 = features['sensors'][sensor]['peak_heights']
    return draw_line(x0, x1, y0, y1, color=color)


def draw_full(df, features, sensor, color):
    x0 = df.index[int(features['sensors'][sensor]['left_ips_full'])]
    x1 = df.index[int(features['sensors'][sensor]['right_ips_full'])]
    y0 = df[sensor].loc[x0]
    y1 = df[sensor].loc[x1]
    return draw_line(x0, x1, y0, y1, color=color)


def draw_half(df, features, sensor, color):
    x0 = df.index[int(features['sensors'][sensor]['left_ips_half'])]
    x1 = df.index[int(features['sensors'][sensor]['right_ips_half'])]
    y0 = df[sensor].loc[x0]
    y1 = df[sensor].loc[x1]
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
