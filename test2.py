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

# Melt the dataframe to "long" format
df_melted = pd.melt(df, id_vars=['sample'], value_vars=['measurement_1', 'measurement_2', 'measurement_3', 'measurement_4'], var_name='measurement')

# Create the line plot
fig = px.line(df_melted, x='sample', y='value', color='measurement')

# Create the scatter plot
fig.add_trace(px.scatter(df_melted, x='sample', y='value', color='measurement', hover_data=['measurement']).data[0])

# Show the plot
fig.show()
