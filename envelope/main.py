import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from read import ReadFile



data, time = ReadFile.read('data.csv')

# Generate noisy 2D data
# t = np.linspace(0, 1, 100)
# x = np.sin(10 * np.pi * t) + 0.2 * np.random.randn(100)
# y = np.sin(5 * np.pi * t) + 0.2 * np.random.randn(100)

y = data
t= time
# Calculate upper envelope
peaks, _ = find_peaks(y)
print(len(peaks))
print(len(data))
# upper_envelope = np.interp(t, t[peaks], y[peaks])

# Plot results
plt.plot(t, y, label='Data')
plt.plot(peaks, data[peaks], label='Noisy Data')
# plt.plot(t, upper_envelope, label='Upper Envelope')
# plt.legend()
plt.show()



