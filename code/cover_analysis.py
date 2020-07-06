#imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import glob

# LOCAL FILES - TO REMOVE LATER
DIR = r"C:\Users\govaerts.kristof\OneDrive - SESVanderHave N.V\Documents\Kristof_phenotyping\CoverAnalysis"
FILE = r"X1Y5 CHAMOIS.JPG"

HUE_RANGE = (80, 90)
ROW_WIDTH = 120
AXIS = 1

#FUNCTIONS
def find_gaps(rowmask, thresh=0.3):
    rowprof = savgol_filter(np.mean(rowmask, axis=0), 101, 3)
    gapf = np.vectorize(lambda x: 1 if x<0.3 else 0)
    gaps = gapf(rowprof)
    return [i for i in range(len(gaps)) if gaps[i]==1]

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
                self.rows[k]["cover"] = 100*np.mean(ss)

    def calc_row_gaps(self):
        for k in self.rows.keys():
            r = self.rows[k]
            if AXIS == 1:
                ss = self.green[int(r["min"]):int(r["max"]), :]
                self.rows[k]["gap_inds"] = find_gaps(ss)

    def rows_figure(self, name=""):
        fig = plt.figure(figsize=(13,10))
        ax = plt.axes()
        ax.imshow(self.bw, cmap="gray")
        ax.imshow(self.green, alpha=0.3)
        ax.set_xlim([0, self.shape[1]+500]) # extra space for annotations
        for k in self.rows.keys():
            r = self.rows[k]
            gap_pos = [r["peak"] for i in range(len(r["gap_inds"]))]
            ax.scatter(r["gap_inds"], gap_pos, s=5, c="r", marker="s")
            ax.plot((1, self.shape[AXIS]), (r["min"], r["min"]), 'b--')
            ax.plot((1, self.shape[AXIS]), (r["max"], r["max"]), 'b--')
            ann = "row: {}\n%cover: {:.3}\n%gaps:  {:.3}".format(k, r["cover"],
                                                100*len(r["gap_inds"])/self.shape[1])
            ax.annotate(ann, xy = (self.shape[1]+100, r["max"]))
        avcov = 100*np.mean(self.green)
        rowcov = np.mean([r["cover"] for r in self.rows.values()])
        ax.set_title("{}: Average cover: {:.3}%, average row cover: {:.3}%".format(name,avcov, rowcov))
        self.fig = fig

    def plot_row_greenness(self):
        for r in self.rows.values():
            row = self.green[r["min"]:r["max"],:]
            rowprof = np.round(savgol_filter(np.mean(row, axis=0), 101, 3))
            plt.plot(rowprof)

os.chdir(DIR)
os.mkdir("test")
filelist = glob.glob("*.JPG")
for f in filelist:
    print(f)
    di = droneImg(os.path.join(DIR, f))
    di.mask(HUE_RANGE)
    di.find_rows()
    di.calc_row_cover()
    di.calc_row_gaps()
    di.rows_figure(name=f)
    di.fig.savefig(os.path.join(DIR, os.path.join("test", f[:-4] + ".png")))