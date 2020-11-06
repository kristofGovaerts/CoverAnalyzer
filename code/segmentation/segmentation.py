import cv2
import numpy as np
from scipy.signal import savgol_filter
from skimage import filters


def filling(im, ker, it=1):
    # fills small holes in image
    i = cv2.dilate(im, ker, iterations=it)
    i = cv2.erode(i, ker, iterations=it)
    return i


def ratio_img(im, ax):
    # Calculate the representation of a certain channel relative to other channels for each pixel.
    return im[:, :, ax]/np.mean(im, axis=2)


def find_gaps(rowmask, thresh=0.2):
    # Find gaps based on the profile along a row. Thresh is the cover fraction for each point along the Y-axis, default
    # is 0.3 ie less than 30% of pixels for this Y-coordinate are green.
    rowprof = savgol_filter(np.mean(rowmask, axis=0), 101, 3)
    gapf = np.vectorize(lambda x: 1 if x<thresh else 0)
    gaps = gapf(rowprof)
    return [i for i in range(len(gaps)) if gaps[i]==1]


def otsu_mask(im, kernel=(3,3)):
    # Use otsu thresholding to find pixels that are overrepresented in the green channel.
    av = ratio_img(im, ax=1)
    av = cv2.GaussianBlur(av, kernel, 0)
    t = filters.threshold_otsu(av)
    av[av < t] = 0
    av[av >= t] = 1
    av = filling(av, np.ones(kernel, np.uint8))
    return av