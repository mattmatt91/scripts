import pandas as pd
import numpy as np
import json as js


def extract_features(data: pd.Series, n_stabw:int) -> dict:
    threshold = np.std(data[:0.2])*n_stabw
    if data.max() >= threshold:
        peak_x = data.idxmax()
        peak_y = data.max()
        base = get_baseline(data, threshold, peak_x)
        middle = get_baseline(data, peak_y/2, peak_x)

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


def get_first_element(data: list, threshold: float, g_s: str, len_offset: int):
    if g_s == 'g':
        for i in range(len(data)):
            if data[i] > threshold:
                return i + len_offset
        return len(data)

    elif g_s == 's':
        for i in range(len(data)):
            if data[i] < threshold:
                return i + len_offset
        return len(data)


def get_baseline(data: pd.Series, heigth: float, peak_x: float):
    data_list = data.to_list()
    index_peak = data.index.get_loc(peak_x)
    data_after_peak = data_list[index_peak:]
    data_before_peak = data_list[:index_peak]
    x1_i = get_first_element(data_before_peak, heigth, 'g', 0)
    x2_i = get_first_element(data_after_peak, heigth,
                             's', len(data_before_peak))
    x1 = data.index[x1_i]
    x2 = data.index[x2_i]
    y1 = heigth
    y2 = heigth
    width = x2 - x1
    # print({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'width': width})
    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'width': width}


def get_integral(data: pd.Series, x1: float, x2: float) -> float:
    try:
        peak = data.loc[x1:x2]
        x = peak.index
        y = peak.to_list()
        return np.trapz(y=y, x=x)
    except:
        return 0


def get_slope(x2: float, y2: float, x1: float, y1: float) -> float:
    slope = (y2 - y1)/(x2*1.001 - x1)
    return slope
