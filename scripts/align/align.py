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

    outlist.append(np.mean(rot, axis=1))  # save intensity profile

o = []
out = np.zeros(int(len(outlist[0])/2))

for i in outlist:
    f = fft(i)
    absrange = 2.0 / len(i) * np.abs(f[0:len(i) // 2]) # get representation of all frequencies
    #o.append(absrange[22]) # position 22 is most common - amount of rows

    der = np.gradient(np.gradient(absrange[5:])) # 2nd derivative
    o.append((np.max(der), 5+np.argmax(der)))
    o.append(np.argmin(np.gradient(np.gradient(absrange[5:]))) + 5)
    out += absrange

plt.imshow(imutils.rotate(green, 47))
