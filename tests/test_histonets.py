#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_histonets
----------------------------------

Tests for `histonets` module.
"""
import os
import unittest

import cv2
import numpy as np

import histonets
from histonets import utils


def image_path(img):
    return os.path.join('tests', 'images', img)


class TestHistonets(unittest.TestCase):
    def setUp(self):
        self.image = cv2.imread(image_path('test.png'))

    def test_adjust_high_contrast(self):
        image = self.image
        image_high_contrast = histonets.adjust_contrast(image, 150)
        test_high_contrast = cv2.imread(image_path('test_high_contrast.png'))
        assert np.array_equal(image_high_contrast, test_high_contrast)

    def test_adjust_low_contrast(self):
        image = self.image
        image_low_contrast = histonets.adjust_contrast(image, 50)
        test_low_contrast = cv2.imread(image_path('test_low_contrast.png'))
        assert np.array_equal(image_low_contrast, test_low_contrast)

    def test_contrast_value_parsing(self):
        image = self.image
        assert np.array_equal(
            histonets.adjust_contrast(image, -10),
            histonets.adjust_contrast(image, 0)
        )
        assert np.array_equal(
            histonets.adjust_contrast(image, 210),
            histonets.adjust_contrast(image, 200)
        )

    def test_lower_brightness(self):
        image = self.image
        image_low_brightness = histonets.adjust_brightness(image, 50)
        test_low_brightness = cv2.imread(
            image_path('test_brightness_darken.png'))
        assert np.array_equal(image_low_brightness, test_low_brightness)

    def test_higher_brightness(self):
        image = self.image
        image_high_brightness = histonets.adjust_brightness(image, 150)
        test_high_brightness = cv2.imread(
            image_path('test_brightness_lighten.png'))
        assert np.array_equal(image_high_brightness, test_high_brightness)

    def test_brightness_value_parsing(self):
        image = self.image
        assert np.array_equal(
            histonets.adjust_brightness(image, -10),
            histonets.adjust_brightness(image, 0)
        )
        assert np.array_equal(
            histonets.adjust_brightness(image, 210),
            histonets.adjust_brightness(image, 200)
        )

    def test_smooth_image(self):
        image = self.image
        smooth_image = histonets.smooth_image(image, 50)
        test_smooth_image = cv2.imread(image_path('smooth50.png'))
        assert np.array_equal(smooth_image, test_smooth_image)

    def test_smooth_image_value_parsing(self):
        image = self.image
        test_smooth100_image = cv2.imread(image_path('smooth100.png'))
        test_smooth0_image = cv2.imread(image_path('smooth0.png'))
        assert np.array_equal(
            histonets.smooth_image(image, 150),
            test_smooth100_image
            )
        assert np.array_equal(
            histonets.smooth_image(image, -50),
            test_smooth0_image
            )

    def test_histogram_equalization(self):
        image = self.image
        test_hist_eq = cv2.imread(image_path('test_hist_eq5.png'))
        assert np.array_equal(
            histonets.histogram_equalization(image, 50),
            test_hist_eq
        )

    def test_histogram_equalization_value_parsing(self):
        image = self.image
        test_hist_eq0 = cv2.imread(image_path('test_hist_eq0.png'))
        test_hist_eq10 = cv2.imread(image_path('test_hist_eq10.png'))
        assert np.array_equal(
            histonets.histogram_equalization(image, -10),
            test_hist_eq0
        )
        assert np.array_equal(
            histonets.histogram_equalization(image, 140),
            test_hist_eq10
        )

    def test_denoise_image(self):
        image = self.image
        test_denoise_img = cv2.imread(image_path('denoised10.png'))
        assert np.array_equal(
            histonets.denoise_image(image, 10),
            test_denoise_img
        )

    def test_denoise_image_value_parsing(self):
        image = self.image
        test_denoise_img0 = cv2.imread(image_path('denoised0.png'))
        test_denoise_img100 = cv2.imread(image_path('denoised100.png'))
        assert np.array_equal(
            histonets.denoise_image(image, -10),
            test_denoise_img0
        )
        assert np.array_equal(
            histonets.denoise_image(image, 110),
            test_denoise_img100
        )

    def test_posterization_linear_4_colors(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_linear4.png'))
        assert np.array_equal(
            histonets.color_reduction(image, 4, 'linear'),
            test_image
        )

    def test_posterization_linear_10_colors(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_linear10.png'))
        assert np.array_equal(
            histonets.color_reduction(image, 10, 'linear'),
            test_image
        )

    def test_posterization_linear_invalid_colors_lower(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_linear2.png'))
        assert np.array_equal(
            histonets.color_reduction(image, 0, 'linear'),
            test_image
        )

    def test_posterization_linear_invalid_colors_higher(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_linear128.png'))
        assert np.array_equal(
            histonets.color_reduction(image, 500, 'linear'),
            test_image
        )

    def test_posterization_kmeans_4_colors(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_kmeans4.png'))
        reduce_image = histonets.color_reduction(image, 4, 'kmeans')
        assert (len(utils.get_palette(test_image))
                == len(utils.get_palette(reduce_image)))

    def test_posterization_kmeans_10_colors(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_kmeans10.png'))
        reduce_image = histonets.color_reduction(image, 10, 'kmeans')
        assert (len(utils.get_palette(test_image))
                == len(utils.get_palette(reduce_image)))

    def test_posterization_kmeans_invalid_colors_lower(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_kmeans2.png'))
        reduce_image = histonets.color_reduction(image, 0, 'kmeans')
        assert (len(utils.get_palette(test_image))
                == len(utils.get_palette(reduce_image)))

    def test_posterization_kmeans_invalid_colors_higher(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_kmeans128.png'))
        reduce_image = histonets.color_reduction(image, 500, 'kmeans')
        assert (len(utils.get_palette(test_image))
                == len(utils.get_palette(reduce_image)))
