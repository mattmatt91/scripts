import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from read import ReadFile



data, time = ReadFile.read('data.csv')

duration = time[-1]
# duration = 1.0

fs = 100000
# fs = 400.0

samples = int(fs*duration)
# print(f'samples: {samples}')

t = time
# t = np.arange(samples) / fs
# print(f't: {t}')

signal = data
# signal = chirp(t, 20.0, t[-1], 100.0)
# print(f'signal: {signal}')

# signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
# print(f'signal: {signal}')


analytic_signal = hilbert(signal)
# print(f'analytic_signal: {analytic_signal}')

amplitude_envelope = np.abs(analytic_signal)
# print(f'amplitude_envelope: {amplitude_envelope}')

# instantaneous_phase = np.unwrap(np.angle(analytic_signal))
# print(f'instantaneous_phase: {instantaneous_phase}')

# instantaneous_frequency = (np.diff(instantaneous_phase) /
                           # (2.0*np.pi) * fs)
# print(f'instantaneous_frequency: {instantaneous_frequency}')

# exit()

fig, ax = plt.subplots()

ax.plot(t, signal, label='signal')

ax.plot(t, amplitude_envelope, label='envelope')

ax.set_xlabel("time in seconds")

ax.legend()

# ax1.plot(t[1:], instantaneous_frequency)

# ax1.set_xlabel("time in seconds")

# ax1.set_ylim(0.0, 120.0)

fig.tight_layout()

plt.show()



