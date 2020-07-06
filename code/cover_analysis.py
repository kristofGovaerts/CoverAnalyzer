#imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

# LOCAL FILES - TO REMOVE LATER
DIR = r"C:\Users\govaerts.kristof\OneDrive - SESVanderHave N.V\Documents\Kristof_phenotyping\CoverAnalysis"
FILE = r"X1Y5 CHAMOIS.JPG"

HUE_RANGE = (80, 90)
ROW_WIDTH = 120
AXIS = 1

#FUNCTIONS

class droneImg:
    def __init__(self, file_loc):
        self.rgb = cv2.imread(file_loc)
        self.hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        self.bw = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        self.shape = self.bw.shape

    def mask(self, thresh):
        f = np.vectorize(lambda x: 1 if min(thresh) <= x <= max(thresh) else 0)
        self.green = f(self.hsv[:, :, 0])

    def find_rows(self, ax=1, sep=50, thresh=1, invert=True):
        av = np.mean(self.bw, axis=AXIS)
        av = savgol_filter(av, 101, 3)  # filter with window length 101 & order 3 - can be very smooth
        if invert:
            av = -av + np.max(av)  # invert because grass is less intense than soil
        peaks = find_peaks(av, height=np.mean(av) - 1.5 * np.std(av), distance=sep)[0]
        self.rows = {str(i): {"peak": int(peaks[i]),
                              "min": int(peaks[i] - ROW_WIDTH/2),
                              "max": int(peaks[i] + ROW_WIDTH/2)} for i in range(len(peaks))}

    def calc_row_cover(self):
        for k in self.rows.keys():
            r = self.rows[k]
            if AXIS == 1:
                ss = self.green[int(r["min"]):int(r["max"]), :]
                self.rows[k]["cover"] = np.mean(ss)


    def show_rows(self):
        plt.imshow(self.bw)
        for k in self.rows.keys():
            r = self.rows[k]
            plt.plot((1, self.shape[AXIS]), (r["peak"], r["peak"]), 'r-')
            plt.plot((1, self.shape[AXIS]), (r["min"], r["min"]), 'b--')
            plt.plot((1, self.shape[AXIS]), (r["max"], r["max"]), 'b--')

    def plot_row_greenness(self):
        for r in self.rows.values():
            row = self.green[r["min"]:r["max"],:]
            rowprof = np.round(savgol_filter(np.mean(row, axis=0), 101, 3))
            plt.plot(rowprof)




di = droneImg(os.path.join(DIR, FILE))
di.mask(HUE_RANGE)
di.find_rows()
di.calc_row_cover()
#di.show_rows()
di.plot_row_greenness()