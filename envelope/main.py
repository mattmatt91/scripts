import numpy as np
import matplotlib.pyplot as plt
from read import ReadFile
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial



data, time = ReadFile.read('data.csv')

def calc_envelope(data, time):
    x_peaks, _ = find_peaks(data, prominence=0.3, distance=50)
    y_peaks = data[x_peaks]
    
    
    plt.plot(data)
    plt.plot(x_peaks, y_peaks)
    plt.plot(x_peaks, data[x_peaks], "o")
    plt.show()


def polynomial_fit(x, y, degree):
    p = Polynomial.fit(x, y, degree)
    return p


calc_envelope(data, time)