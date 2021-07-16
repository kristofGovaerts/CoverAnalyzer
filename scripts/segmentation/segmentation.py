import cv2
import numpy as np
from skimage import filters
from scripts.analysis.maths import vari


def filling(im, ker, it=1):
    # fills small holes in image
    i = cv2.dilate(im, ker, iterations=it)
    i = cv2.erode(i, ker, iterations=it)
    return i


def ratio_img(im, ax):
    # Calculate the representation of a certain channel relative to other channels for each pixel.
    return im[:, :, ax]/np.mean(im, axis=2)


def otsu_mask(im, kernel=(3, 3)):
    # Use otsu thresholding to find pixels that are overrepresented in the green channel.
    av = ratio_img(im, ax=1)
    av = cv2.GaussianBlur(av, kernel, 0)
    t = filters.threshold_otsu(av[~np.isnan(av)])
    av[av < t] = 0
    av[av >= t] = 1
    av = filling(av, np.ones(kernel, np.uint8))
    return av


def vari_mask(bgrim, kernel=(3,3)):
    v = vari(bgrim)
    v[v > 0] = 1
    v[v <= 0] = 0
    v = filling(v, np.ones(kernel, np.uint8))
    return v
