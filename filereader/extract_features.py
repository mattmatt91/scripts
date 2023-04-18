import pandas as pd
import numpy as np
import json as js


def extract_features(data: pd.Series, threshold: float) -> dict:
    if data.max() >= threshold: # and data.idxmax() < 0.01 and data.idxmax() > 0.0008:
        peak_x = data.idxmax()
        peak_y = data.max()

        base = get_baseline(data, threshold)
        middle = get_baseline(data, peak_y/2)

        slope_pos = get_slope(
            x1=base['x1'],
            y1=base['y1'],
            x2=peak_x,
            y2=peak_y)

        slope_neg = get_slope(
            x1=peak_x,
            y1=peak_y,
            x2=base['x2'],
            y2=base['y2'])

        integral = get_integral(data, base['x1'], base['x2'])

        # features beginnning with _ are deleted after plotitng, no influence on statistics
        result = {
            '_threshold': threshold,
            '_peak_x': peak_x,
            'peak_y': peak_y,
            '_base_x1': base['x1'],
            'base_y1': base['y1'],
            '_base_x2': base['x2'],
            'base_y2': base['y2'],
            'base_width': base['width'],
            '_half_x1': middle['x1'],
            'half_y1': middle['y1'],
            '_half_x2': middle['x2'],
            'half_y2': middle['y2'],
            'half_width': middle['width'],
            'integral': integral,
            'slope_pos': slope_pos,
            'slope_neg': slope_neg
        }
        return result
    else:
        return {}


def get_baseline(data: pd.Series, heigth: float):
    data_new = data[data > heigth]
    # print(data_new.index.to_list())
    ix1 = data.index.get_loc(data_new.index[0])
    x1 = data.index[ix1-1]
    y1 = data[x1]

    if data_new.index[-1] == data.index[-1]:
        x2 = data.index[-1]
    else:
        ix2 = data.index.get_loc(data_new.index[-1])
        x2 = data.index[ix2+1]
    y2 = data[x2]
    # print(heigth, x2)
    width = x2-x1

    # print({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'width': width})
    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'width': width}


def get_integral(data: pd.Series, x1: float, x2: float) -> float:
    peak = data.loc[x1:x2]
    x = peak.index
    y = peak.to_list()
    return np.trapz(y=y, x=x)


def get_slope(x2: float, y2: float, x1: float, y1: float) -> float:
    slope = (y2 - y1)/(x2 - x1)
    return slope
