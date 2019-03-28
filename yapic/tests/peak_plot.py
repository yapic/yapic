from tkinter import ttk, N, S, HORIZONTAL
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import gridspec
import numpy as np
import seaborn as sns
from peakfinder.views import config as cf


class PeakPlot(ttk.Frame):

    def __init__(self, master, controller):
        super().__init__(master)
        self.c = controller
        self.create_widgets()
        self.c.roi_change_cb_funcs.append(self.update_fig)
        self.c.peak_change_cb_funcs.append(self.update_fig)
        self.c.label_change_cb_funcs.append(self.update_fig)
        self.c.predict_cb_funcs.append(self.update_fig)
        self.c.relative_frame_change_cb_funcs.append(self.update_fig)

    def create_widgets(self):
        self.init_fig()
        self.cv = FigureCanvasTkAgg(self.fig, master=self)
        self.cv.draw()
        self.canvas = self.cv.get_tk_widget()
        self.canvas.config(height=50, width=1000, scrollregion=(0, 0, 0, 100))
        self.canvas.grid(row=0, column=0, sticky=(N, S))

        self.fig.set_size_inches(100/100,
                                 1000/100)

    def roi_change_cb(self, *args):
        self.update_fig()

    def init_fig(self):
        # fig, (ax, ax1) = plt.subplots(2, 1)
        fig = plt.figure(facecolor=(240/255., 240/255., 237/255., 1), dpi=100)
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], wspace=0, hspace=0)
        with sns.axes_style('whitegrid'):
            ax1 = fig.add_subplot(gs[0])
        with sns.axes_style('white'):
            ax2 = fig.add_subplot(gs[1])

        ax1.axis('off')
        ax2.axis('off')
        ax1.margins(x=0, y=0)
        ax2.margins(x=0, y=0)

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig.subplots_adjust(left=0., right=1, top=1, bottom=0.)

    def update_fig(self, *args):

        self.ax1.cla()
        self.ax2.cla()
        data = self.c.roidata
        if data is not None:

            ymin = np.min(data)
            ymax = np.max(data)

            self.ax1.plot(data, color='c')

            peaks = self.c.get_active_peaks(label=1)
            if peaks is not None:
                times_lbl_valid = [p.time for p in peaks]
                self.ax1.plot(times_lbl_valid, [ymax+3] * len(times_lbl_valid),
                              'vg')

            peaks = self.c.get_active_peaks(label=-1)
            if peaks is not None:
                times_lbl_valid = [p.time for p in peaks]
                self.ax1.plot(times_lbl_valid, [ymax+3] * len(times_lbl_valid),
                              'vr')

            peaks = self.c.get_active_peaks(cls=1, label=0)
            if peaks is not None:
                times_lbl_valid = [p.time for p in peaks]
                self.ax1.plot(times_lbl_valid, [ymax+3] * len(times_lbl_valid),
                              'v', color=cf.lightgreen)

            peaks = self.c.get_active_peaks(cls=-1, label=0)
            if peaks is not None:
                times_lbl_valid = [p.time for p in peaks]
                self.ax1.plot(times_lbl_valid, [ymax+3] * len(times_lbl_valid),
                              'v', color=cf.lightred)

            peak = self.c.get_active_peak()
            if peak is not None:
                time = peak.time + self.c.relative_frame.get()
                self.ax1.plot([time, time], [ymin, ymax], 'k')
                peaktime = peak.time
                self.ax1.plot([peaktime, peaktime], [ymin, ymax], 'r')

            self.ax2.pcolor(np.expand_dims(data, axis=0))
            self.ax1.axis('off')
            self.ax2.axis('off')
            self.ax1.margins(x=0, y=0)
            self.ax2.margins(x=0, y=0)

            self.cv.draw()
