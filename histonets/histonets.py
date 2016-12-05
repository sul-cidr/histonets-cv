# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import exposure

from .utils import image_as_array


@image_as_array
def adjust_contrast(image, contrast):
    if (contrast < 0):
        contrast = 0
    elif (contrast > 200):
        contrast = 200
    contrast = contrast / 100
    img = image.astype(np.float) * contrast
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.ubyte)


@image_as_array
def adjust_brightness(image, brightness):
    if (brightness < 0):
        brightness = 0
    elif (brightness > 200):
        brightness = 200
    brightness = (((brightness) * (510)) / 200) - 255
    img = image.astype(np.float) + brightness
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.ubyte)


@image_as_array
def smooth_image(image, kernel):
    if (kernel < 0):
        kernel = 0
    elif (kernel > 100):
        kernel = 100
    return cv2.bilateralFilter(image, kernel, kernel, kernel)


@image_as_array
def histogram_equalization(image, tile):
    # scale from Javi's code is 0 to 8
    if (tile < 0):
        tile = 0
    elif (tile > 8):
        tile = 8
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2 ** tile, 2 ** tile))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
    img = exposure.rescale_intensity(img_out)
    return img
