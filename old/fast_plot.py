import pandas as pd
from os import path, listdir, makedirs
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt
import json


# direcotrys
path_data = 'C:\\Users\\49157\\Desktop\\Paper II\\data\\22.09.2022_test'
path_result = 'C:\\Users\\49157\\Desktop\\Paper II\\results'

# plot params
length_plot_x_axis = 0.15
pre_length_plot_x_axis = 0.02

# quickview
quickview = True
measurement = '14-30-02' # sippet identifying measurements


#read measurements
def read_measurements(path_data):
    # list all measurements
    subfolder = [path.join(path_data, i) for i in listdir(path_data)]

    # iterate over all subfolders
    for folder in subfolder:
        print(folder)
        filenames = [path.join(folder, i) for i in listdir(folder)]
        for file in filenames:
            print(file)
            # if file.find('.json')>0:
            #     info = read_json(file)
            #     print(info, 5)
            if file.find('.txt')>0:
                data = pd.read_csv(file, delimiter='\t', decimal='.')
                data.set_index('time [s]', inplace=True)
                data = clean_data(data)
                if quickview and file.find(measurement)>=0: # only plot 
                    plot_stacked(data, file)
                elif not quickview: # crop and save
                    data = data.apply(lambda x: floating_mean(x), axis=0) # floating mean
                    plot_stacked(data, file)


# plot stacked plot 
def plot_stacked(data, file):
    name = file[file.rfind('\\')+1:file.rfind('.txt')]
    fig = px.line(data, y=data.columns, title=name)
    fig.update_layout(
        yaxis_title="Voltage [V]",
        legend_title="Sensors",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        ))
    if quickview: # only showing
        fig.show()
    else: # only saving
        folder_name = path_data[path_data.rfind('\\')+1:]
        path_save = path.join(path_result,folder_name, 'plots')
        makedirs(path_save, exist_ok=True)
        path_save_file_html = path.join(path_save, name+'.html')
        path_save_file_png = path.join(path_save, name+'.png')
        fig.write_html(path_save_file_html)
        fig.write_image(path_save_file_png)


def clean_data(data):
    means = data[0:1].mean() # set zero point
    data = data - means 
    flag = True
    data = data.abs() # set all positive
    
    i = 0
    while flag: #cut relevant section
        index_to_cut = data[data['Piezo1'].gt(1)].index[i]  
        if data['Piezo1'].iloc[data.index.get_loc(index_to_cut)+1] > 1:
            flag = False
        else:
            i +=1    
    data = data[index_to_cut - pre_length_plot_x_axis: index_to_cut + length_plot_x_axis]
    data = data.apply(lambda x: floating_mean(x), axis=0) # floating mean
    return data


def floating_mean(data):
    N = 25
    data_flat = np.convolve(data, np.ones(N)/N, mode='valid')
    data = pd.Series(data_flat, index=data.index[:len(data_flat)], name= data.name)
    return data

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    read_measurements(path_data)