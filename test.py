import plotly.graph_objs as go
from plotly.subplots import make_subplots

layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000,
)

fig = make_subplots(rows=3, cols=1)  # , start_cell="bottom-left")
fig1 = go.Scatter(x=[1, 2, 3], y=[7, 8, 9])
fig1b = go.Scatter(x=[0, 1, 2, 0], y=[0, 2, 0, 0], fill="toself")
fig2 = go.Scatter(x=[1, 2, 3], y=[7, 8, 9])
fig3 = go.Scatter(x=[1, 2, 3], y=[7, 8, 9])

figs = [fig1b, fig1, fig2, fig3]


rows = [1, 1, 2, 3]
cols = [1, 1, 1, 1]
# for f in data:
fig.add_traces(figs, cols=cols, rows=rows)
# fig = go.Figure(data=data)
fig.show()
