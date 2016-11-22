# -*- coding: utf-8 -*-
import numpy as np


def adjust_contrast(image, contrast):
    if (contrast < -100):
        contrast = 0
    elif (contrast > 100):
        contrast = 100
    contrast = (contrast + 100) / 100
    img = image.astype(np.float) * contrast
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.ubyte)
