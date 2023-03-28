import pandas as pd
import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths
import json as js


def extract_features(data: pd.Series, threshold: float) -> dict:
    peaks, peak_properties = find_peaks(
        data, prominence=0, width=1, distance=20000, height=threshold)

    if len(peaks) > 0:
        results_full = peak_widths(data, peaks, rel_height=0.99)
        values_full = [i[0] for i in np.array(results_full)]
        results_half = peak_widths(data, peaks, rel_height=0.5)
        values_half = [i[0] for i in np.array(results_half)]
        peak_info = {
            # peak_info["peak_time"] = data.index[peaks[0]]
            # peak_info["prominences"] = peak_properties["prominences"][0]
            "peak": data[data.index[peaks[0]]],
            "widths": peak_properties["widths"][0],
            "width_heights": peak_properties["width_heights"][0],
            "peak_heights": peak_properties["peak_heights"][0],
            "width_full": values_full[0],
            "width_half": values_half[0],
            "width_height_full": values_full[1],
            "width_height_half": values_half[1],
            "integral": get_integral(data, peak_properties),
            "slope": get_slope(data, peak_properties, peaks),
            'plot_info': {'start_f': data.index[int(peak_properties['left_bases'][0])],
                          'stop_f': data.index[int(peak_properties['right_bases'][0])],
                          'start_h': data.index[int(values_half[2])],
                          'stop_h': data.index[int(values_half[3])],
                          'val': data[data.index[peaks[0]]],
                          'tmax': data.index[peaks[0]]}
        }
        # print(js.dumps(peak_info, sort_keys=True, indent=4))
        return peak_info
    else:
        return {}


def get_integral(data: pd.Series, peak_properties: dict) -> float:
    end = peak_properties['right_bases'][0]
    start = peak_properties['left_bases'][0]
    t_end = data.index[end]
    t_start = data.index[start]
    peak = data[t_start:t_end]
    x = peak.index
    y = peak.to_list()
    return np.trapz(y=y, x=x)


def get_slope(data: pd.Series, peak_properties: dict, peaks: dict) -> float:
    start_time = data.index[peak_properties['left_bases'][0]]
    start = data[start_time]
    stop_time = data.index[peaks[0]]
    stop = data[data.index[peaks[0]]]
    slope = (stop - start)/(stop_time - start_time)
    return slope
