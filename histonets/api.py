# -*- coding: utf-8 -*-
import sys
import warnings
from collections import namedtuple

import cv2
import noteshrink
import numpy as np
import PIL
from imutils import object_detection
from skimage import exposure
from skimage import feature
from skimage import morphology
from skimage import filters

from .utils import (
    convert,
    image_as_array,
    get_palette,
    kmeans,
    match_template_mask,
    get_quantize_method,
    output_as_mask,
    sample_histogram,
)


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
def color_reduction(image, n_colors, method='kmeans', palette=None):
    """Reduce the number of colors in image to n_colors using method"""
    method = method.lower()
    if method not in ('kmeans', 'linear', 'max', 'median', 'octree'):
        method = 'kmeans'
    if n_colors < 2:
        n_colors = 2
    elif n_colors > 128:
        n_colors = 128
    if method == 'kmeans':
        n_clusters = n_colors
        h, w = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        img = img.reshape((-1, 3))  # -1 -> img.shape[0] * img.shape[1]
        centers, labels = kmeans(img, n_clusters)
        if palette is not None:
            # palette comes in RGB
            centers = cv2.cvtColor(np.array([palette]), cv2.COLOR_RGB2LAB)[0]
        quant = centers[labels].reshape((h, w, 3))
        output = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    else:
        img = PIL.Image.fromarray(image[:, :, ::-1], mode='RGB')
        quant = img.quantize(colors=n_colors,
                             method=get_quantize_method(method))
        if palette is not None:
            palette = np.array(palette, dtype=np.uint8)
            quant.putpalette(palette.flatten())
        output = np.array(quant.convert('RGB'), dtype=np.uint8)[:, :, ::-1]
    return output


@image_as_array
def auto_clean(image, background_value=25, background_saturation=20,
               colors=8, sample_fraction=5, white_background=False,
               saturate=True, palette=None):
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
    Options = namedtuple(
        'options',
        ['quiet', 'sample_fraction', 'value_threshold', 'sat_threshold']
    )
    options = Options(
        quiet=True,
        sample_fraction=sample_fraction / 100.0,
        value_threshold=background_value / 100.0,
        sat_threshold=background_saturation / 100.0,
    )
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if palette is None:
        samples = noteshrink.sample_pixels(rgb_image, options)
        palette = get_palette(samples, colors, background_value,
                              background_saturation)
    labels = noteshrink.apply_palette(rgb_image, palette, options)
    if saturate:
        palette = palette.astype(np.float32)
        pmin = palette.min()
        pmax = palette.max()
        palette = 255 * (palette - pmin) / ((pmax - pmin) or 1)
        palette = palette.astype(np.uint8)
    if white_background:
        palette = palette.copy()
        palette[0] = (255, 255, 255)
    return palette[labels]


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


@image_as_array
def color_mask(image, color, tolerance=0):
    """Extract a mask of image according to color under a certain
    tolerance level (defaults to 0)."""
    if tolerance > 100:
        tolerance = 100
    elif tolerance < 0:
        tolerance = 0
    tolerance = int(tolerance * 255 / 100)
    red, green, blue = color
    bgr_color = np.uint8([[[blue, green, red]]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
    mask_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_range = hsv_color - np.array([tolerance, 0, 0])
    lower_range[lower_range > 255] = 255
    lower_range[lower_range < 0] = 0
    upper_range = hsv_color + np.array([tolerance, 0, 0])
    upper_range[upper_range > 255] = 255
    upper_range[upper_range < 0] = 0
    mask = cv2.inRange(mask_image, lower_range, upper_range)
    return mask


@image_as_array
@output_as_mask
def select_colors(image, colors, return_mask=False):
    """Apply several masks to image, each for a color and tolerance, returning
    the result.

    Each entry in colors is a tuple with an RGB tuple representing the color
    and a value for the tolerance from 0 to 100: ((123, 45, 98), 10)."""
    mask = False
    for color, tolerance in colors:
        mask |= color_mask(image, color, tolerance)
    return image, mask


@image_as_array
@output_as_mask
def remove_ridges(image, width=6, threshold=160, dilation=1,
                  return_mask=False):
    """Detect ridges of width pixels using the highest eigenvector of the
    Hessian matrix, then create a binarized mask with threshold and remove
    it from image (set to black). Default values are optimized for text
    detection and removal.

    A dilation radius in pixels can be passed in to thicken the mask prior
    to being applied."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The value of sigma is calculated according to Steger's work:
    # An Unbiased Detector of Curvilinear Structures,
    # IEEE Transactions on Pattern Analysis and Machine Intelligence,
    # Vol. 20, No. 2, Feb 1998
    # http://ieeexplore.ieee.org/document/659930/
    sigma = (width / 2) / np.sqrt(3)
    hxx, hxy, hyy = feature.hessian_matrix(gray_image, sigma=sigma, order='xy')
    large_eigenvalues, _ = feature.hessian_matrix_eigvals(hxx, hxy, hyy)
    mask = convert(large_eigenvalues)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    if dilation:
        dilation = (2 * dilation) + 1
        dilation_kernel = np.ones((dilation, dilation), np.uint8)
        mask = cv2.dilate(mask, dilation_kernel)
    return image, 255 - mask


@image_as_array
@output_as_mask
def remove_blobs(image, min_area=0, max_area=sys.maxsize, threshold=128,
                 method='8-connected'):
    """Binarize image using threshold, and remove (turn into black)
    blobs of connected pixels of white of size bigger or equal than
    min_area but smaller or equal than max_area from the original image,
    returning it afterward."""
    method = method.lower()
    if method == '4-connected':
        method = cv2.LINE_4
    elif method in ('16-connected', 'antialiased'):
        method = cv2.LINE_AA
    else:  # 8-connected
        method = cv2.LINE_8
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mono_image = cv2.threshold(gray_image, threshold, 255, 0)
    _, all_contours, _ = cv2.findContours(mono_image, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array([contour for contour in all_contours
                         if min_area <= cv2.contourArea(contour) <= max_area])
    mask = np.ones(mono_image.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, 0, -1, lineType=method)
    return image, mask


@image_as_array
def binarize_image(image, method='li', **kwargs):
    """Binarize image using one of the available methods: 'isodata',
    'li', 'otsu', and 'sauvola'. Defaults to 'li'. Extra keyword arguments are
    passed in as is to the corresponding sciki-image thresholding function.
    For reference
    Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding Techniques
    and Quantitative Performance Evaluation" Journal of Electronic Imaging,
    13(1): 146-165 DOI:10.1117/1.1631315
    """
    if image.ndim != 2:
        # image is not gray-scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.unique(image).size == 2:
        # image is already binary
        return image
    if method not in ('sauvola', 'isodata', 'otsu', 'li'):
        method = 'li'
    thresh_func = getattr(filters.thresholding, "threshold_{}".format(method))
    threshold = thresh_func(image, **kwargs)
    # OpenCV can't write black and white images using boolean values, it needs
    # at least a 8bits 1-channel image ranged from 0 (black) to 255 (white)
    return convert(image <= threshold)


@image_as_array
def skeletonize_image(image, method=None, dilation=None, binarization=None):
    """Extract a 1 pixel wide representation of image by morphologically
    thinning the white contiguous blobs (connected components).
    If image is not black and white, a binarization process is applied
    according to binarization, which can be 'sauvola', 'isodata', 'otsu',
    'li' (default, ref: binarize()).

    A process of dilation can also be applied by specifying the number
    of pixels in dilate prior to extracting the skeleton.

    The method for skeletonization can be 'medial', '3d', 'regular', or a
    'combined' version of the three. Defaults to 'regular'.
    A 'thin' operator is also available. For reference,
    see http://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
    """
    # scikit-image needs only 0's and 1's
    mono_image = binarize_image(image, method=binarization) / 255
    if dilation:
        dilation = (2 * dilation) + 1
        dilation_kernel = np.ones((dilation, dilation), np.uint8)
        mono_image = cv2.morphologyEx(mono_image, cv2.MORPH_DILATE,
                                      dilation_kernel)
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings('ignore', category=UserWarning)
        if method == '3d':
            skeleton = morphology.skeletonize_3d(mono_image)
        elif method == 'medial':
            skeleton = morphology.medial_axis(mono_image,
                                              return_distance=False)
        elif method == 'thin':
            skeleton = morphology.thin(mono_image)
        elif method == 'combined':
            skeleton = (morphology.skeletonize_3d(mono_image)
                        | morphology.medial_axis(mono_image,
                                                 return_distance=False)
                        | morphology.skeletonize(mono_image))
        else:
            skeleton = morphology.skeletonize(mono_image)
    return convert(skeleton)


def histogram_palette(histogram, n_colors=8, method='auto', sample_fraction=5,
                      background_value=25, background_saturation=20):
    """Return a palette of at most n_colors unique colors extracted
    after sampling histogram by sample_fraction."""
    sampled_histogram = sample_histogram(histogram,
                                         sample_fraction=sample_fraction)
    return get_palette(
        sampled_histogram, method=method,
        n_colors=n_colors, background_value=background_value,
        background_saturation=background_saturation)
