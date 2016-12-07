# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import exposure
from sklearn.cluster import MiniBatchKMeans

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
    if (tile < 0):
        tile = 0
    elif (tile > 100):
        tile = 100
    tile = int(tile / 10)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2 ** tile, 2 ** tile))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
    img = exposure.rescale_intensity(img_out)
    return img


@image_as_array
def denoise_image(image, value):
    if (value < 0):
        value = 0
    elif (value > 100):
        value = 100
    return cv2.fastNlMeansDenoisingColored(image, None, value, value)


@image_as_array
def color_reduction(image, n_colors, method='kmeans'):
    """Reduce the number of colors in image to n_colors using method"""
    method = method.lower()
    if method not in ('kmeans', 'linear'):
        method = 'kmeans'
    if n_colors < 2:
        n_colors = 2
    elif n_colors > 128:
        n_colors = 128
    if method == 'kmeans':
        n_clusters = n_colors
        h, w = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        clf = MiniBatchKMeans(n_clusters=n_clusters)
        labels = clf.fit_predict(img)
        raw_quant = clf.cluster_centers_.astype(np.ubyte)[labels]
        quant = raw_quant.reshape((h, w, 3))
        reduced = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    else:
        # convert to the exponent of the next power of 2
        n_levels = int(np.ceil(np.log(n_colors) / np.log(2)))
        indices = np.arange(0, 256)
        divider = np.linspace(0, 255, n_levels + 1)[1]
        quantiz = np.int0(np.linspace(0, 255, n_levels))
        color_levels = np.clip(np.int0(indices / divider), 0, n_levels - 1)
        palette = quantiz[color_levels]
        reduced = cv2.convertScaleAbs(palette[image])
    return reduced
