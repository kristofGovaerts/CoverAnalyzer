# WIP code for automatic image rotation

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from tqdm import tqdm  # for progress bar
from scipy.signal import find_peaks, savgol_filter
from scripts.segmentation.segmentation import ratio_img


def estimate_peaks(samples, thresh=0.02):
    """Get the height of the peaks in a signal vs. the mean of the signal without those pks. If this value is high (>1,
    the signal has strong, defined peaks."""
    pks = find_peaks(samples, prominence=thresh)
    npks = np.ones(len(samples), dtype=bool)
    npks[pks[0]] = False
    return np.mean(samples[pks[0]])/np.mean(samples[npks]), len(pks[0])


def assess_rotations(im, step=1, channel=1, thresh=0.02):
    """
    Produces the 1-dimensional intensity profile along an axis of the image, for each rotation between 0 and 179 degrees
    :param im: An RGB image. 2-dimensional with 3 channels.
    :param step: The step size for the iteration. Smaller step sizes are more accurate, but it takes longer to perform
    all iterations.
    :param channel: An int between 0 and 2. 0=red, 1=green, 2=blue.
    :param thresh: A value between 0 and 1. Peak threshold.
    :return: A dict containing intensity profiles, peak prominences and peak numbers.
    """
    green = ratio_img(im, channel)
    angles = np.arange(180, step=step)
    outlist = []
    qs = []
    ns = []

    print("Evaluating rotational angles...")

    for i in tqdm(range(len(angles))):
        a=angles[i]
        rot = imutils.rotate(green, a)
        rot[rot==0] = np.nan
        r = np.nanmean(rot, axis=1)

        outlist.append(r)  # save intensity profile

        r[np.isnan(r)] = 0
        filtered = r[np.argwhere(r)][:, 0]
        q, n = estimate_peaks(filtered, thresh=thresh)
        qs.append(q)
        ns.append(n)
    return {"profiles": outlist,
            "prominences": qs,
            "numbers": ns}


def align_image(rgb, thresh=0.02, step=1, channel=1):
    """
    Aligns the image so that the rows are arranged vertically and the columns horizontally.
    :param rgb: An RGB image. 2-dimensional with 3 channels.
    :param thresh: A threshold between 0 and 1 for the peaks. Lower is better if contrast is low, but rows may be
    incorrectly identified.
    :param step: Step size for assessing rotations. Can in principle be any value <180. The smaller, the more iterations
    are needed, and the more often the input image has to be rotated.
    :param channel: The channel of the RGB image to assess. 0=red, 1=green, 2=blue
    :return: An RGB image and the number of rows identified in this image.
    """
    r = assess_rotations(rgb, thresh=thresh, step=step, channel=channel)
    rot_id = np.argmax(r['prominences'])
    best_rotation = step * rot_id
    n_rows = r['numbers'][rot_id]
    print("Optimal rotation angle: {} degrees.".format(best_rotation))
    print("Number of rows identified: {}.".format(n_rows))
    rotated_image = imutils.rotate(rgb, best_rotation)
    return rotated_image


if __name__=='__main__':
    file_loc = r'C:\Users\Kristof\Dropbox\Python\CoverAnalyzer\examples\alignment\test2.JPG'
    rgb = cv2.imread(file_loc)

    thresh=0.02
    step=1
    channel=1

    r = assess_rotations(rgb, thresh=thresh, step=step, channel=channel)
    rot_id = np.argmax(r['prominences'])
    best_rotation = step * rot_id
    n_rows = r['numbers'][rot_id]
    print("Optimal rotation angle: {} degrees.".format(best_rotation))
    print("Number of rows identified: {}.".format(n_rows))
    rotated_image = imutils.rotate(rgb, best_rotation)

    samples = len(r['profiles'][0])

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 20))
    axs[0, 0].set_title('Raw image')
    axs[0, 0].imshow(rgb)

    axs[0, 1].set_title('Intensity profile')
    axs[0, 1].plot(r['profiles'][0], np.arange(samples))
    axs[0, 1].invert_yaxis()

    axs[1, 0].set_title('Optimally rotated image')
    axs[1, 0].imshow(imutils.rotate(rgb, best_rotation))

    axs[1, 1].set_title('Intensity profile')
    axs[1, 1].plot(r['profiles'][best_rotation], np.arange(samples))
    axs[1, 1].invert_yaxis()

    axs[2, 0].set_title('Optimal rotation + 90degrees')
    axs[2, 0].imshow(imutils.rotate(rgb, best_rotation+90))

    axs[2, 1].set_title('Intensity profile')
    axs[2, 1].plot(r['profiles'][best_rotation+90], np.arange(samples))
    axs[2, 1].invert_yaxis()

    plt.show()