import pandas as pd
import numpy as np
import json as js


def extract_features(data: pd.Series, threshold: float) -> dict:
    if data.max() >= threshold and data.idxmax() < 0.01 and data.idxmax() > 0.0008:
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
    # exit()
    ix1 = next(n for n in range(len(data.index)) if data.iloc[n] > heigth)
    x1 = data.index[ix1-1]
    y1 = data.loc[x1]

    data_2nd = data.iloc[ix1:]
    index = data_2nd[data_2nd<heigth].index
    if len(index)>0:
        x2 = index[0]
    else:
        x2 = data.index[len(data.index)-1]
    y2 = data.loc[x2]

    width = x2-x1
    print( {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'width': width})
    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'width': width}


def get_integral(data: pd.Series, x1: float, x2: float) -> float:
    peak = data.loc[x1:x2]
    x = peak.index
    y = peak.to_list()
    return np.trapz(y=y, x=x)


def get_slope(x2: float, y2: float, x1: float, y1: float) -> float:
    slope = (y2 - y1)/(x2 - x1)
    return slope



import numpy as np
import pandas as pd

def calculate_intersections(data, line_height):
    # Convert the Pandas DataFrame to a NumPy array for easier indexing
    data_arr = data.to_numpy()

    # Calculate the number of rows and columns in the data array
    num_rows, num_cols = data_arr.shape

    # Initialize an empty list to store the intersections
    intersections = []

    # Loop over each column in the data array
    for j in range(num_cols):
        # Initialize a variable to store the index of the row where the line intersects the data
        intersection_idx = -1

        # Loop over each row in the data array
        for i in range(num_rows):
            # If the data value is less than or equal to the line height and the previous data value is greater than the line height,
            # the line intersects the data between these two rows
            if i > 0 and data_arr[i, j] <= line_height and data_arr[i-1, j] > line_height:
                # Calculate the y value of the intersection
                y_intersection = line_height

                # Calculate the x value of the intersection using linear interpolation
                x1 = i - 1
                x2 = i
                y1 = data_arr[i-1, j]
                y2 = data_arr[i, j]
                x_intersection = x1 + ((y_intersection - y1) / (y2 - y1)) * (x2 - x1)

                # Add the intersection to the list of intersections
                intersections.append((x_intersection, y_intersection))

    # Convert the list of intersections to a NumPy array and return it
    return np.array(intersections)
