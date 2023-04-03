import numpy as np
import matplotlib.pyplot as plt
from read import ReadFile
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial


data, time = ReadFile.read('data.csv')


def calc_envelope(data, time, threshold: float):
    N = 100
    new_data = np.convolve(data, np.ones(N)/N, mode='valid')
    x_peaks, _ = find_peaks(new_data, distance=100)
    
    y_peaks = data[x_peaks]

    # print(peak_data)F
    plt.plot(x_peaks, y_peaks)
    plt.plot(x_peaks, data[x_peaks], "o")
    plt.show()


calc_envelope(data, time, 0.008)
