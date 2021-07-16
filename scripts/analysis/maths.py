import numpy as np


def vari(bgrim):
    """Visible Atmospherically Resistant Index = (G-R)/(G+R-B)"""
    a = np.array(bgrim).astype(np.float)
    r = a[:, :, 2]
    g = a[:, :, 1]
    b = a[:, :, 0]
    vari = (g-r)/(g+r-b)
    vari[vari < -1] = -1
    vari[vari > 1] = 1
    return vari
