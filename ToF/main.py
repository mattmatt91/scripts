import pandas as pd
import plotly.express as px
import numpy as np

def read_file_and_plot(path:str, name:str):
    df = pd.read_csv(path, delimiter=',', decimal='.')
    df.set_index('Cycle', inplace=True)
    cols_to_plot = [i for i in df.columns if i.find('Raw')>= 0]
    fig = px.line(df[cols_to_plot], title=name)
    fig.show()
    return df

def integrate_and_normalise(df:pd.DataFrame):
    data = {}
    my_val = 0
    for col in df.columns:
        val = np.trapz(df[col])
        data[col] = val
        my_val += val
    print(data)
    print(my_val)








path_traces_new = "E:\\PTR-TOF\\HMTD Analysis\\Exports\\Neu\\2023-06-14_12-57-59 HMTD_neu - traces.csv"
path_traces_old = "E:\\PTR-TOF\\HMTD Analysis\\Exports\\Alt\\2023-07-26 HMTD_alt traces.csv"


df_new = read_file_and_plot(path_traces_new, 'new')
df_old = read_file_and_plot(path_traces_old, 'old')
integrate_and_normalise(df_new)





exit()



path = "E:\\PTR-TOF\\HMTD Analysis\\Exports\\Alt\\Spectra_Old.csv"
df_spec_old = pd.read_csv(path, delimiter=',', decimal='.')
print(df_spec_old)

df_spec_old.set_index(df_spec_old.columns[0], inplace=True)


fig = px.line(df_spec_old, title='Aged HMTD', y='Intensities')
fig.show()

path = "E:\\PTR-TOF\\HMTD Analysis\\Exports\\Neu\\Spectra_New.csv"
df_spec_new = pd.read_csv(path, delimiter=',', decimal='.')
print(df_spec_new)

df_spec_new.set_index(df_spec_new.columns[0], inplace=True)


fig = px.line(df_spec_new, title='New HMTD', y='Intensities')
fig.show()