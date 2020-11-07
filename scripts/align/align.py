# WIP code for automatic image rotation

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from scipy.fft import fft

from scipy.signal import find_peaks, savgol_filter
from scripts.segmentation.segmentation import ratio_img


file_loc = r'C:\Users\Kristof\Dropbox\Python\CoverAnalyzer\examples\alignment\test2.JPG'
rgb = cv2.imread(file_loc)
green = ratio_img(rgb, 1)

#test all rotations between 1deg and 180deg
angles = [1 + i for i in range(180)]
outlist = []
for a in angles:
    rot = imutils.rotate(green, a)
    rot[rot==0] = np.nan

    outlist.append(np.nanmean(rot, axis=1))  # save intensity profile

qs = []
ns = []


def estimate_peaks(samples, thresh=0.02):
    """Get the height of the peaks in a signal vs. the mean of the signal. If this value is high (>1,
    the signal has strong, defined peaks."""
    pks = find_peaks(samples, prominence=thresh)
    return np.mean(samples[pks[0]])/np.mean(samples), len(pks[0])


for i in outlist:
    i[np.isnan(i)] = 0
    filtered = i[np.argwhere(i)][:,0]
    q, n = estimate_peaks(filtered)
    qs.append(q)
    ns.append(n)

best_rotation = np.argmax(qs)

plt.imshow(imutils.rotate(green, best_rotation))
print("number of peaks found: {}".format(ns[best_rotation]))