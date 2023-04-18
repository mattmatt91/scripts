import numpy as np
import pandas as pd

class PreProcessing:
    def cut_time_section(data: pd.DataFrame, properties: dict) -> pd.DataFrame:
        sensor_to_cut = properties['sensor_to_cut']
        threshold = properties['sensors'][sensor_to_cut]['threshold']
        cut_before_signal = properties['cut_before_signal']
        cut_after_signal = properties['cut_after_signal']
        index = data[sensor_to_cut].idxmax()
        data = data.iloc[index-cut_before_signal:index+cut_after_signal]
        if data[sensor_to_cut].max() > threshold:
            return data
        else:
            raise ValueError("no peak in vis")


    def remove_offset(data: pd.DataFrame, properties: dict) -> pd.DataFrame:
        means = data[0:properties['points_offset']].mean()  # set zero point
        data = data - means
        std = np.std(data[0:properties['points_offset']])*properties['n_std']
        return data


    def smooth_and_abs_data(data, properties):
        for sensor in data:
            if properties['sensors'][sensor]['abs']:
                data[sensor] = data[sensor].abs()
            if type(properties['sensors'][sensor]['smooth']) == int:
                # data[sensor] = PreProcessing.smooth_me(data[sensor])
                data[sensor] = PreProcessing.floating_mean(
                    data[sensor], n=properties['sensors'][sensor]['smooth'])
        return data


    def create_time_axis(data: pd.DataFrame, info: dict) -> pd.DataFrame:
        data['time'] = np.round(np.arange(start=0, stop=len(
            data)*(1/info['rate']), step=1/info['rate'], dtype=float), 5)
        data.reset_index(drop=True, inplace=True)
        data.set_index('time', inplace=True)
        return data


    def floating_mean(data:pd.DataFrame, n=25):
        data_flat = np.convolve(data, np.ones(n)/n, mode='same')
        data = pd.Series(
            data_flat, index=data.index[:len(data_flat)], name=data.name)
        return data
    
    def smooth_me( df: pd.DataFrame)->pd.DataFrame:
        last_value = 0
        smoothed = []
        for i in range(len(df)):
            if df.iloc[i] < df.iloc[i:i+1].max():
                smoothed.append(last_value)
            else:
                last_value = df.iloc[i]
                smoothed.append(last_value) 
        return smoothed