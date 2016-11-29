# -*- coding: utf-8 -*-
import cv2
import numpy as np


def adjust_contrast(image, contrast):
    if (contrast < -100):
        contrast = -100
    elif (contrast > 100):
        contrast = 100
    contrast = (contrast + 100) / 100
    img = image.astype(np.float) * contrast
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.ubyte)


def adjust_brightness(image, brightness):
    if (brightness < -100):
        brightness = -100
    elif (brightness > 100):
        brightness = 100
    brightness = (((brightness + 100) * 51) / 20) - 255
    img = image.astype(np.float) + brightness
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.ubyte)


def smooth_image(image, kernel):
    if (kernel < 0):
        kernel = 0
    elif (kernel > 100):
        kernel = 100
    return cv2.bilateralFilter(image, kernel, kernel, kernel)
