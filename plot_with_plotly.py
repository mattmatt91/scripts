import plotly.express as px
import pandas as pd



def plot_stacked(data):
    fig = px.line(data, x=data.index, y=data.columns)
    fig.show()