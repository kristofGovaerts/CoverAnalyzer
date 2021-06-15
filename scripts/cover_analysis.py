# Automatic batch tool for assessing cover on RGB drone images in .JPG format.

# imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy import optimize

import glob
from tkinter import filedialog
import pandas as pd
from sympy import symbols, solve

from scripts.segmentation.segmentation import otsu_mask, ratio_img
from scripts.analysis.analysis import cosine_func, calc_row_cover, calc_row_gaps
from scripts.align.align import align_image

# GLOBAL VARS
OUTPUT_FOLDER = "output" # The name of the output folder.
ROW_WIDTH = 30  # row width in pixels - change if necessary
# TODO: Automatic row width detection

ROW_FINDING = "periodic"  # 'periodic' or 'automatic'
AXIS = 1  # does not work yet, will implement if necessary
KERNEL = (3, 3)  # for masking - larger values = more blurry but less noisy mask
THRESH = 0.2
ALIGN = True


class DroneImg:
    def __init__(self, file_loc):
        self.rgb = cv2.imread(file_loc)
        self.bw = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        self.file_location = file_loc
        self.shape = self.rgb.shape
        # placeholders
        self.green = None
        self.rows = None
        self.fig = None
        self.window_length = int(ROW_WIDTH+int(not (ROW_WIDTH % 2)))  # window length has to be odd

    def mask(self):
        self.green = otsu_mask(self.rgb, kernel=KERNEL)

    def align(self):
        aligned = align_image(self.rgb)
        out_filename = self.file_location.replace('.', '_aligned.')
        print("Writing aligned image to location: {}".format(out_filename))
        cv2.imwrite(out_filename, aligned)
        self.rgb = aligned
        self.bw = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)

    def find_rows(self, sep=ROW_WIDTH/2):
        # Take the intensity profile along the X-axis for green pixels and use scipy.signal's find_peaks to identify
        # peak positions
        av = ratio_img(self.rgb, ax=1)
        av = np.nanmean(av, axis=1)
        av = savgol_filter(av, self.window_length, 3)  # filter with window length based on sep & order 3

        peaks = find_peaks(av, height=np.nanmean(av), distance=sep)[0]
        self.rows = {str(i): {"peak": int(peaks[i]),
                              "min": int(peaks[i] - ROW_WIDTH/2),
                              "max": int(peaks[i] + ROW_WIDTH/2)} for i in range(len(peaks))}

    def find_rows2(self):
        # use a cosinusoidal fit to determine row positions. Works quite well for this type of data and ensures peaks
        # are evenly spaced, making it more robust to signal fluctuations or flat peaks
        # x and y arrays
        ysig = np.nanmean(ratio_img(self.rgb, ax=1), axis=1)
        ysig = 2*(ysig-min(ysig))/max(ysig-min(ysig)) - 1  # normalize to -1-1, ie basic cosine func
        xsig = np.arange(len(ysig))

        # starting values: a = amplitude, b = period (x-units), c = phase (x-units), d = y-offset
        self.find_rows()
        row_no = len(self.rows)
        print("Estimated row number: {}".format(row_no))
        b0 = len(xsig) / row_no
        c0 = 2.0
        # d0 = np.mean(ysig)

        # fit parameters
        p, pv = optimize.curve_fit(cosine_func, xsig, ysig, [b0, c0],
                                   bounds=([0, -np.inf], [np.inf, np.inf]))
        per = p[0]  # period in x-units
        # calculate x-offset of maximum: max of cosine function is reached when (x/b + 1/c)=1. solve() equals to 0
        pha = solve((symbols('x') / p[0]) + (1 / p[1]) - 1)[0]
        while pha < 0:
            pha += per  # determine starting point
        peaks = [pha + i*per for i in range(row_no)]
        self.rows = {str(i): {"peak": int(peaks[i]),
                              "min": int(peaks[i] - ROW_WIDTH/2),
                              "max": int(peaks[i] + ROW_WIDTH/2)} for i in range(len(peaks))}

    def calc_row_cover(self):
        # Calculate average cover for each row.
        self.rows = calc_row_cover(self.rows, self.green)

    def calc_row_gaps(self):
        # Calculate gap postitions.
        self.rows = calc_row_gaps(self.rows, self.green, self.window_length, THRESH)

    def rows_figure(self, name=""):
        fig, ax = plt.subplots(1, 1, figsize=(13, 10))

        ax.imshow(self.bw, cmap="gray")
        ax.imshow(self.green, alpha=0.3)
        ax.set_xlim([0, self.shape[1]+500])  # extra space for annotations
        for k in self.rows.keys():
            r = self.rows[k]
            gap_pos = [r["peak"] for i in range(len(r["gap_inds"]))]
            ax.scatter(r["gap_inds"], gap_pos, s=5, c="r", marker="s")
            ax.plot((1, self.shape[AXIS]), (r["min"], r["min"]), 'b--')
            ax.plot((1, self.shape[AXIS]), (r["max"], r["max"]), 'b--')
            ann = "row: {}\n%cover: {:.3}\n%gaps:  {:.3}".format(k, r["cover"],
                                                                 100*len(r["gap_inds"])/self.shape[1])
            ax.annotate(ann, xy=(self.shape[1]+100, r["max"]))
        avcov = 100*np.nanmean(self.green)
        rowcov = np.nanmean([r["cover"] for r in self.rows.values()])
        ax.set_title("{}: Average cover: {:.3}%, average row cover: {:.3}%".format(name, avcov, rowcov))

        self.fig = fig


def main():
    d = filedialog.askdirectory()
    os.chdir(d)

    try:
        os.mkdir(OUTPUT_FOLDER)
    except FileExistsError:
        print("Warning: Target directory /{}/ already found. Files may be overwritten."
              .format(os.path.join(d, OUTPUT_FOLDER)))

    filelist = glob.glob("*.JPG")
    outlist = []

    for f in filelist:
        print(f)
        p = os.path.join(d, f)
        di = DroneImg(p)

        if ALIGN:
            di.align()

        di.mask()
        cv2.imwrite(os.path.join(d, os.path.join(OUTPUT_FOLDER, f[:-4] + "_mask.png")), 255*di.green)

        if ROW_FINDING == 'automatic':
            print("Trying to automatically determine row positions.")
            di.find_rows()
        elif ROW_FINDING == 'periodic':
            print("Trying to use a sinusoidal fit to determine row positions.")
            di.find_rows2()
        else:
            print("Provided row finding method not understood.")
            break

        print("Rows found: {}".format(len(di.rows)))
        di.calc_row_cover()
        di.calc_row_gaps()
        di.rows_figure(name=f)
        di.fig.savefig(os.path.join(d, os.path.join(OUTPUT_FOLDER, f[:-4] + ".png")))

        avgap = np.nanmean([100*len(v["gap_inds"])/di.shape[1] for v in di.rows.values()])
        out = [f, 100*np.nanmean(di.green), np.nanmean([v["cover"] for v in di.rows.values()]), avgap, len(di.rows.keys())]
        outlist.append(out)

    outfile = pd.DataFrame(outlist)
    outfile.columns = ["filename", "total_cover", "av_row_cover", "av_gaps", "rows"]
    outfile.to_csv(os.path.join(OUTPUT_FOLDER, "cover_statistics.csv"), sep='\t', index=False)


if __name__ == "__main__":
    print("Executing CoverAnalysis script.")
    main()
