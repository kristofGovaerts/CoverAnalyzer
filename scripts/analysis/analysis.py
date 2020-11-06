import numpy as np


def cosine_func(x, b, c):
    # 2-factor sine function
    # b = period (x-units), c = phase (x-units)
    return np.cos((2*np.pi*x)/b + 2*np.pi/c)
