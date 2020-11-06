import numpy as np
from scipy.signal import savgol_filter


def cosine_func(x, b, c):
    # 2-factor sine function
    # b = period (x-units), c = phase (x-units)
    return np.cos((2*np.pi*x)/b + 2*np.pi/c)


def calc_row_cover(rows: dict, green: np.ndarray):
    """
    Calculate average cover for each row.

    :param rows: A dictionary containing the positions and min/max of the rows in the image.
    :param green: The accompanying intensity image of the green channel.
    :return: An updated dictionary with added cover info.
    """
    out = dict(rows)

    for k in rows.keys():
        r = rows[k]
        ss = green[int(r["min"]):int(r["max"]), :]
        out[k]["cover"] = 100*np.mean(ss)
    return out


def find_gaps(rowmask, window_length, thresh=0.2):
    """
    Find gaps based on the profile along a row. Thresh is the cover fraction for each point along the Y-axis, default
    is 0.3 ie less than 30% of pixels for this Y-coordinate are green.

    :param rowmask: An 2D array containing intensity data from one row.
    :param window_length: An int.
    :param thresh: A float between 0 and 1 indicating the cover threshold for a point being part of a gap.
    :return: Gap pixel positions along the length of the row.
    """

    rowprof = savgol_filter(np.mean(rowmask, axis=0), window_length, 3)
    gapf = np.vectorize(lambda x: 1 if x<thresh else 0)
    gaps = gapf(rowprof)
    return [i for i in range(len(gaps)) if gaps[i]==1]


def calc_row_gaps(rows: dict, green: np.ndarray, window_length: int, thresh: float):
    """
    Calculate gap indices for each row.

    :param rows: A dictionary containing the positions and min/max of the rows in the image.
    :param green: The accompanying intensity image of the green channel.
    :param window_length: An int specifying window length.
    :param thresh: A float between 0 and 1 indicating the cover threshold for a point being part of a gap.
    :return: An updated dictionary with added gap info.
    """
    out = dict(rows)

    for k in rows.keys():
        r = rows[k]

        if r["min"] < 0:
            mi = 0
        else:
            mi = int(r["min"])

        if r["max"] >= green.shape[0] - 1:
            ma = green.shape[0] - 1
        else:
            ma = int(r["max"])

        ss = green[mi:ma, :]
        out[k]["gap_inds"] = find_gaps(ss, window_length=window_length, thresh=thresh)
    return out
