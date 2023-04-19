import plotly.express as px
import pandas as pd

# Create a sample dataframe with four measurements per sample
df = pd.DataFrame({
    'sample': ['A', 'B', 'C', 'D', 'E', 'F'],
    'measurement_1': [1, 2, 3, 4, 5, 6],
    'measurement_2': [2, 3, 4, 5, 6, 7],
    'measurement_3': [3, 4, 5, 6, 7, 8],
    'measurement_4': [4, 5, 6, 7, 8, 9]
})


df2 = pd.DataFrame({
    'sample': ['A', 'B', 'C', 'D', 'E', 'F'],
    'measurement_21': [11, 24, 23, 14, 15, 16],
    'measurement_22': [21, 31, 14, 15, 16, 17],
    'measurement_23': [31, 41, 51, 61, 71, 8],
    'measurement_24': [24, 25, 26, 27, 28, 29]
})

# Melt the dataframe to "long" format
df_melted = pd.melt(df, id_vars=['sample'], value_vars=['measurement_1', 'measurement_2', 'measurement_3', 'measurement_4'], var_name='measurement')
df_melted2 = pd.melt(df2, id_vars=['sample'], value_vars=['measurement_12', 'measurement_22', 'measurement_32', 'measurement_42'], var_name='measurement')

# Create the line plot
fig = px.line()

# Create the scatter plot
fig.add_trace(px.scatter(df_melted, x='sample', y='value', color='measurement', hover_data=['measurement']).data[0])
fig.add_trace(px.scatter(df_melted2, x='sample', y='value', color='measurement', hover_data=['measurement']).data[0])

# Show the plot
fig.show()
