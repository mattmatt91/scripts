import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_measurement(info, df, peak_properties, results_half, results, full, peaks):
    Plot()
class Plot:
    def __init__(self,name, size, properties):
        self.plot_properties = properties['plot_properties']['measurement_plot']
        self.properties = properties
        self.fig, self.axs = plt.subplots(size, sharex=True, dpi=self.plot_properties['dpi'], figsize=self.plot_properties['size'])
        self.name = name
        self.i = 0

    def add_subplot(self, sensor, df, peak_properties, results_half, results_full, peaks):
        self.axs[self.i].plot(df[sensor], color=self.properties['sensors'][sensor]['color'])
        ## print peaks in plot
        if peaks.size != 0:
            self.axs[self.i].plot(df.index[peaks], df[sensor][df.index[peaks]], "x")
            self.axs[self.i].vlines(x=df.index[peaks][0], ymin=df[sensor][df.index[peaks][0]] - peak_properties["prominences"],
                       ymax=df[sensor][df.index[peaks][0]], color="C1")
            self.axs[self.i].hlines(y=peak_properties["width_heights"], xmin=df.index[int(peak_properties["left_ips"])],
                       xmax=df.index[int(peak_properties["right_ips"])], color="C1")
            self.axs[self.i].hlines(y=results_full[1], xmin=df.index[int(results_full[2])],
                       xmax=df.index[int(results_full[3])],
                       color="C2")
            self.axs[self.i].hlines(y=results_half[1], xmin=df.index[int(results_half[2])],
                       xmax=df.index[int(results_half[3])],
                       color="C2")

        label = sensor + ' [V]'
        self.axs[self.i].set_ylabel(label, rotation=0, loc='top', fontsize = self.plot_properties['label_size'])
        self.axs[self.i].tick_params(axis='y', labelsize= self.plot_properties['font_size'])
        self.axs[self.i].grid()
        try:
            self.axs[self.i].set_yticks(np.arange(0,np.max(df[sensor]),round(np.max(df[sensor])/3, 2)))
        except:
            self.axs[self.i].set_yticks(np.arange(0,5,5/3))
        self.i = self.i +1

    def show_fig(self, path):
        self.axs[-1].set_xlabel("time [s]" , fontsize = self.plot_properties['label_size'])
        plt.xticks(fontsize=self.plot_properties['font_size'])
        self.axs[-1].get_shared_x_axes().join(*self.axs)
        self.fig.tight_layout()
        path = path + '\\results\\plots\\single_measurements'
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + '\\' + self.name + '.jpeg'
        
        self.fig.tight_layout()
        # plt.show()
        self.fig.savefig(path)
        plt.close(self.fig)