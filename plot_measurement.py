import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import plotly.express as px
from helpers import mkdir_ifnotexits
from os.path import join
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_measurement(df, peak_data, properties, name, path):
    fig = px.line(df, x=df.index, y=df.columns)


    for i, sensor in zip(range(len(df.columns)), df.columns):
        color = properties['sensors'][fig.data[i].name]['color']
        fig.data[i].line.color = color
        if len(peak_data[sensor]['peaks']) > 0:
            fig = draw_peak(fig, df, peak_data, sensor, color)
    # fig.show()
    
    path=f"{path}\\results\\plots\\measurements\\combined"
    mkdir_ifnotexits(path)
    path=join(path, f'{name}.html')
    fig.write_html(path)


def draw_peak(fig, df, peak_data, sensor, color):
    fig = draw_v(fig, df, peak_data, sensor, color)
    # fig = draw_h(fig, df, peak_data, sensor, color)
    fig = draw_half(fig, df, peak_data, sensor, color)
    fig = draw_full(fig, df, peak_data, sensor, color)
    return fig

def draw_v(fig, df, peak_data, sensor, color):
    x0 = df.index[peak_data[sensor]['peaks'][0]]
    x1 = x0
    y0 = 0
    y1 = peak_data[sensor]['peak_properties']['peak_heights'][0]
    fig = draw_line(fig, x0, x1, y0, y1, color)
    return fig


def draw_h(fig, df, peak_data, sensor, color):
    x0 = df.index[peak_data[sensor]['peak_properties']['left_bases'][0]]
    x1 = df.index[peak_data[sensor]['peak_properties']['right_bases'][0]]
    y0 = df[sensor].loc[x0]
    y1 = df[sensor].loc[x1]
    fig = draw_line(fig, x0, x1, y0, y1, color)
    return fig

def draw_full(fig, df, peak_data, sensor, color):
    keys = "widths heigth left_ips right_ips".split()
    values = [i[0] for i in np.array(peak_data[sensor]['results_full'])]
    results_full = {}
    for key, val in zip(keys, values):
        results_full[key] = val
    x0 =  df.index[int(results_full['left_ips'])]
    x1 =  df.index[int(results_full['right_ips'])]
    y0 = results_full['heigth']
    y1 = results_full['heigth']
    fig = draw_line(fig, x0, x1, y0, y1, color)
    return fig

def draw_half(fig, df, peak_data, sensor, color):
    keys = "widths heigth left_ips right_ips".split()
    values = [i[0] for i in np.array(peak_data[sensor]['results_half'])]
    results_half = {}
    for key, val in zip(keys, values):
        results_half[key] = val
    x0 =  df.index[int(results_half['left_ips'])]
    x1 =  df.index[int(results_half['right_ips'])]
    y0 = results_half['heigth']
    y1 = results_half['heigth']
    fig = draw_line(fig, x0, x1, y0, y1, color)
    return fig

def draw_line(fig, x0, x1, y0, y1, color):
    fig.add_shape(type="line",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(
            color=color,
            width=4,
        )
    )
    return fig

# Piezo2 peaks [300]
# Piezo2 peak_properties {'peak_heights': array([1.49031269]), 'prominences': array([1.49014866]), 'left_bases': array([253], dtype=int64), 'right_bases': array([2114], dtype=int64), 'widths': array([1.04341631]), 'width_heights': array([0.74523836]), 'left_ips': array([299.49432752]), 'right_ips': array([300.53774383])}
# Piezo2 results_full (array([62.87805035]), array([0.01506552]), array([256.01251869]), array([318.89056904]))
# Piezo2 results_half (array([1.04341631]), array([0.74523836]), array([299.49432752]), array([300.53774383]))
# Piezo2 result_dict {'Piezo2_peak[s]': 0.003, 'Piezo2_start[s]': 0.00256, 'Piezo2_stop[s]': 0.00318, 'Piezo2_width[s]': 0.0006199999999999999, 'Piezo2_width_half[s]': 1.0000000000000026e-05, 'Piezo2_height': 1.4903126931494626, 'Piezo2_intetegral[Vs]': 0.0001756448384754551, 'Piezo2_slope[V/s]': array([47690.69356917]), 'Piezo2_width_heigth[s/V]': 0.0004160200760887034}


def plot_measurement_stacked(df, peak_data, properties, name, path):
    titles = tuple(df.columns)
    fig = make_subplots(rows=len(df.columns), cols=1, subplot_titles=titles)
    for col, i in zip(df.columns, range(len(df.columns))):
        color = properties['sensors'][col]['color']
        fig.append_trace(go.Scatter(
            fillcolor=color,
            showlegend=False,
            x=df.index,
            y=df[col],
        ), row=i+1, col=1)
    fig.update_layout(height=1200, width=800, title_text=name)

    path=f"{path}\\results\\plots\\measurements\\stacked"
    mkdir_ifnotexits(path)
    path=join(path, f'{name}_stacked.html')
    fig.write_html(path)

   