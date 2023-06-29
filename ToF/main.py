import pandas as pd
import plotly.express as px
import numpy as np

def read_file_and_plot(path:str, name:str):
    df = pd.read_csv(path, delimiter=',', decimal='.')
    df.set_index('Cycle', inplace=True)
    cols_to_plot = [i for i in df.columns if i.find('Raw')>= 0]
    fig = px.line(df[cols_to_plot], title=name)
    # fig.show()
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




path_traces_new = "C:\\Users\\matth\\Desktop\\PaperII\\data\\PtR-TOF\\Exports\\Neu\\2023-06-14_12-57-59 HMTD_neu - traces.csv"
path_traces_old = "C:\\Users\\matth\\Desktop\\PaperII\\data\\PtR-TOF\\Exports\\Alt\\2023-06-14_13-23-08 HMTD_alt traces.csv"


df_new = read_file_and_plot(path_traces_new, 'new')
df_old = read_file_and_plot(path_traces_old, 'old')
integrate_and_normalise(df_new)












exit()
path = "C:\\Users\\matth\\Desktop\\PaperII\\data\\PtR-TOF\\Exports\\Alt\\2023-06-14_13-23-08 HMTD_alt spectra average.csv"
df_spec = pd.read_csv(path, delimiter=',', decimal='.')
print(df_spec)
df_spec.set_index(df_spec.columns[0], inplace=True)
df_spec.set_index(df_spec.columns[0], inplace=True)

fig = px.line(df_spec)
fig.show()
