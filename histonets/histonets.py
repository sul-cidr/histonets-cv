# -*- coding: utf-8 -*-
from collections import namedtuple

import cv2
import noteshrink
import numpy as np
from imutils import object_detection
from PIL import Image as PILImage
from skimage import exposure

from .utils import image_as_array, get_palette, kmeans, match_template_mask


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
        centers, labels = kmeans(img, n_clusters)
        quant = centers[labels].reshape((h, w, 3))
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


@image_as_array
def auto_clean(image, background_value=25, background_saturation=20,
               colors=8, sample_fraction=5, white_background=False,
               saturate=True):
    """Clean image with minimal input required. Based on the work by
    Matt Zucker: https://mzucker.github.io/2016/09/20/noteshrink.html"""
    if background_value < 1:
        background_value = 1
    elif background_value > 100:
        background_value = 100
    if background_saturation < 1:
        background_saturation = 1
    elif background_saturation > 100:
        background_saturation = 100
    if sample_fraction < 1:
        sample_fraction = 1
    elif sample_fraction > 100:
        sample_fraction = 100
    if colors < 2:
        colors = 2
    elif colors > 128:
        colors = 128
    options = namedtuple(
        'options',
        ['quiet', 'sample_fraction', 'value_threshold', 'sat_threshold']
    )(
        quiet=True,
        sample_fraction=sample_fraction / 100.0,
        value_threshold=background_value / 100.0,
        sat_threshold=background_saturation / 100.0,
    )
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    samples = noteshrink.sample_pixels(rgb_image, options)
    palette = get_palette(samples, colors, background_value,
                          background_saturation)
    labels = noteshrink.apply_palette(rgb_image, palette, options)
    if saturate:
        palette = palette.astype(np.float32)
        pmin = palette.min()
        pmax = palette.max()
        palette = 255 * (palette - pmin) / (pmax - pmin)
        palette = palette.astype(np.uint8)
    if white_background:
        palette = palette.copy()
        palette[0] = (255, 255, 255)
    output = PILImage.fromarray(labels, 'P')
    output.putpalette(palette.flatten())
    return np.array(output.convert('RGB'))[:, :, ::-1]  # swap R and G channels


@image_as_array
def match_templates(image, templates, overlap=0.15):
    """Look for templates in image and return the matches.

    Each entry in the templates list is a dictionary with keys 'image',
    'threshold', 'flip', 'mask' and its matching
    'method' (None, 'laplacian', 'canny')."""
    default_threshold = 80
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectangles = np.empty([0, 2, 2], dtype=int)
    for template in templates:
        threshold = template.get('threshold', default_threshold)
        if threshold > 100:
            threshold = 100
        elif threshold < 0:
            threshold = 0
        threshold /= 100.0
        template_image = template.get('image')
        template_flip = template.get('flip')
        template_mask = template.get('mask')
        template_method = template.get('method', 'canny')  # defaults to canny
        gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        transformations = [lambda im: im]
        if template_flip:
            if template_flip[0] in ('h', 'a'):
                transformations.append(lambda im: cv2.flip(im, 1))
            elif template_flip[0] in ('v', 'a'):
                transformations.append(lambda im: cv2.flip(im, 0))
            elif template_flip[0] in ('b', 'a'):
                transformations.append(lambda im: cv2.flip(cv2.flip(im, 1), 0))
        for transformation in transformations:
            transformed_template = transformation(gray_template)
            height, width = transformed_template.shape
            if template_mask is not None:
                transformed_mask = transformation(template_mask)
            else:
                transformed_mask = None
            results = match_template_mask(gray_image, transformed_template,
                                          transformed_mask, template_method)
            index = results >= threshold
            y1, x1 = np.where(index)
            y2, x2 = y1 + height, x1 + width
            coords = np.array([x1, y1, x2, y2], dtype=int).T
            probs = results[index]
            boxes = np.array(
                object_detection.non_max_suppression(coords, probs, overlap)
            )
            xyboxes = boxes.reshape(boxes.shape[0], 2, 2)  # list of x,y points
            rectangles = np.vstack([rectangles, xyboxes])
    return rectangles.astype(int)
