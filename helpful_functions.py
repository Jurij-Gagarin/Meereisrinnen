# In this module a few simple functions are stored that can be very useful
from math import log10, floor
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def flatten_list(images):
    flat_img = []
    for img in images:
        if isinstance(img, list):
            for sub_img in img:
                flat_img.append(sub_img)
        else:
            flat_img.append(img)
    return flat_img


def filter_by_sic(var1, var2, sic, trigger=90):
    mask = sic < trigger
    return var1[mask], var2[mask], sic[mask]
