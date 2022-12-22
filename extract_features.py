import pandas as pd
import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths


def get_peak(data: pd.Series, threshold: float) -> dict:
    peaks, peak_properties = find_peaks(
        data, prominence=0, width=1, distance=20000, height=threshold)
    if len(peaks) > 0:
        peak_info = {}
        peak_info["peak_time"] = data.index[peaks[0]]
        for key in peak_properties:
            peak_info[key] = peak_properties[key][0] 
        
        

        keys = "widths heigth left_ips right_ips".split()

        results_full = peak_widths(data, peaks, rel_height=0.99)
        values_full = [i[0] for i in np.array(results_full)]
        for key, val in zip(keys, values_full):
            peak_info[f"{key}_full"] = val

        results_half = peak_widths(data, peaks, rel_height=0.5)
        values_half = [i[0] for i in np.array(results_half)]
        for key, val in zip(keys, values_half):
            peak_info[f"{key}_half"] = val

        val1 = int(results_full[2][0])
        val2 = int(results_full[3][0])
        df_peak = pd.DataFrame(data.iloc[val1:val2])
        peak_info['peak'] = df_peak

        return peak_info
    else:
        return {}

def extract_features(data: pd.Series, peak_info: dict):
    dict_feature = {"intetegral[Vs]": get_integral,
                    "slope[V/s]": get_slope
                    }

    features = {}
    for key in dict_feature:
        features[key] = dict_feature[key](data, peak_info)
    peak_info.pop('peak', None)
    features = features | peak_info
    return features


def get_integral(data: pd.Series, peak_info: dict):
    x = peak_info['peak'].index.tolist()
    y = [i[0] for i in peak_info['peak'].values.tolist()]
    # print(x,y)
    return np.trapz(y=y, x=x)


def get_slope(data: pd.Series, peak_info: dict) -> float:
    start = data[data.index[int(peak_info['left_ips_half'])]]
    start_time = data.index[int(peak_info['left_ips_half'])]
    stop = data[peak_info['peak_time']]
    stop_time = peak_info['peak_time']
    slope = (stop_time - start_time)/(stop - start)
    return slope
