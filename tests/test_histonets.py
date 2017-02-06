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
        assert (len(utils.get_color_histogram(test_image))
                == len(utils.get_color_histogram(reduce_image)))

    def test_posterization_kmeans_10_colors(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_kmeans10.png'))
        reduce_image = histonets.color_reduction(image, 10, 'kmeans')
        assert (len(utils.get_color_histogram(test_image))
                == len(utils.get_color_histogram(reduce_image)))

    def test_posterization_kmeans_invalid_colors_lower(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_kmeans2.png'))
        reduce_image = histonets.color_reduction(image, 0, 'kmeans')
        assert (len(utils.get_color_histogram(test_image))
                == len(utils.get_color_histogram(reduce_image)))

    def test_posterization_kmeans_invalid_colors_higher(self):
        image = self.image
        test_image = cv2.imread(image_path('poster_kmeans128.png'))
        reduce_image = histonets.color_reduction(image, 500, 'kmeans')
        assert (len(utils.get_color_histogram(test_image))
                == len(utils.get_color_histogram(reduce_image)))

    def test_auto_clean(self):
        image = self.image
        test_image = cv2.imread(image_path('clean.png'))
        reduce_image = histonets.auto_clean(image)
        assert (len(utils.get_color_histogram(test_image))
                == len(utils.get_color_histogram(reduce_image)))

    def test_auto_clean_non_default(self):
        image = self.image
        test_image = cv2.imread(image_path('clean.png'))
        reduce_image = histonets.auto_clean(
            image, background_value=30, background_saturation=25,
            colors=12, sample_fraction=7, white_background=True,
            saturate=False
        )
        assert (len(utils.get_color_histogram(test_image))
                != len(utils.get_color_histogram(reduce_image)))

    def test_auto_clean_invalid_lower(self):
        image = self.image
        test_image = cv2.imread(image_path('clean1.png'))
        reduce_image = histonets.auto_clean(
            image, background_value=-1, background_saturation=-1,
            colors=-1, sample_fraction=-1
        )
        assert (len(utils.get_color_histogram(test_image))
                == len(utils.get_color_histogram(reduce_image)))

    def test_auto_clean_invalid_higher(self):
        image = self.image
        test_image = cv2.imread(image_path('clean100.png'))
        reduce_image = histonets.auto_clean(
            image, background_value=110, background_saturation=110,
            colors=150, sample_fraction=110
        )
        assert (len(utils.get_color_histogram(test_image))
                == len(utils.get_color_histogram(reduce_image)))

    def test_match_templates(self):
        image = self.image
        template = cv2.imread(image_path('template.png'))
        templates = [
            {'image': template, 'threshold': 95}
        ]
        matches = histonets.match_templates(image, templates)
        test_matches = [((259, 349), (329, 381))]
        assert np.array_equal(test_matches, matches)

    def test_match_templates_thresholds(self):
        image = self.image
        template = cv2.imread(image_path('template.png'))
        templates1 = [
            {'image': template, 'threshold': 95}
        ]
        matches1 = histonets.match_templates(image, templates1)
        templates2 = [
            {'image': template, 'threshold': 5}
        ]
        matches2 = histonets.match_templates(image, templates2)
        assert not np.array_equal(matches1, matches2)

    def test_match_templates_threshold_default(self):
        image = self.image
        template = cv2.imread(image_path('template.png'))
        templates = [
            {'image': template, 'threshold': 80}
        ]
        matches = histonets.match_templates(image, templates)
        default_templates = [
            {'image': template}
        ]
        default_matches = histonets.match_templates(image, default_templates)
        assert np.array_equal(default_matches, matches)

    def test_match_templates_threshold_high(self):
        image = self.image
        template = cv2.imread(image_path('template.png'))
        templates_high = [
            {'image': template, 'threshold': 150}
        ]
        matches_high = histonets.match_templates(image, templates_high)
        templates_100 = [
            {'image': template, 'threshold': 100}
        ]
        matches_100 = histonets.match_templates(image, templates_100)
        assert np.array_equal(matches_high, matches_100)

    def test_match_templates_threshold_low(self):
        image = self.image
        template = cv2.imread(image_path('template.png'))
        templates_low = [
            {'image': template, 'threshold': -10}
        ]
        matches_low = histonets.match_templates(image, templates_low)
        templates_0 = [
            {'image': template, 'threshold': 0}
        ]
        matches_0 = histonets.match_templates(image, templates_0)
        assert np.array_equal(matches_low, matches_0)

    def test_match_templates_flip_horizontally(self):
        image = self.image
        template = cv2.imread(image_path('template.png'))
        templates = [
            {'image': template, 'threshold': 95}
        ]
        matches = histonets.match_templates(image, templates)
        templates_flip = [
            {'image': cv2.imread(image_path('template_h.png')), 'flip': 'h'}
        ]
        matches_flip = histonets.match_templates(image, templates_flip)
        assert np.array_equal(matches_flip, matches)

    def test_match_templates_flip_vertically(self):
        image = self.image
        template = cv2.imread(image_path('template.png'))
        templates = [
            {'image': template, 'threshold': 95}
        ]
        matches = histonets.match_templates(image, templates)
        templates_flip = [
            {'image': cv2.imread(image_path('template_v.png')), 'flip': 'v'}
        ]
        matches_flip = histonets.match_templates(image, templates_flip)
        assert np.array_equal(matches_flip, matches)

    def test_match_templates_flip_both(self):
        image = self.image
        template = cv2.imread(image_path('template.png'))
        templates = [
            {'image': template, 'threshold': 95}
        ]
        matches = histonets.match_templates(image, templates)
        templates_flip = [
            {'image': cv2.imread(image_path('template_b.png')), 'flip': 'b'}
        ]
        matches_flip = histonets.match_templates(image, templates_flip)
        assert np.array_equal(matches_flip, matches)

    def test_match_templates_mask(self):
        image = self.image
        template = cv2.imread(image_path('template_m.png'))
        polygon = [
            [50, 50],
            [120, 50],
            [120, 82],
            [50, 82],
        ]
        mask = utils.get_mask_polygons([polygon], *template.shape[:2])
        templates = [{
            'image': template,
            'threshold': 45,
            'mask': mask,
        }]
        test_matches = [[[209, 299], [379, 431]]]
        matches = histonets.match_templates(image, templates)
        assert np.array_equal(test_matches, matches)

    def test_color_mask(self):
        image = cv2.imread(image_path('poster_kmeans4.png'))
        image_mask = cv2.imread(image_path('mask_tol50.png'), 0)  # B&W
        color = (58, 36, 38)
        tolerance = 50
        mask = histonets.color_mask(image, color, tolerance)
        assert np.array_equal(image_mask, mask)

    def test_color_mask_lower(self):
        image = cv2.imread(image_path('poster_kmeans4.png'))
        image_mask = cv2.imread(image_path('mask_tol0.png'), 0)  # B&W
        color = (58, 36, 38)
        tolerance = -1
        mask = histonets.color_mask(image, color, tolerance)
        assert np.array_equal(image_mask, mask)

    def test_color_mask_higher(self):
        image = cv2.imread(image_path('poster_kmeans4.png'))
        image_mask = cv2.imread(image_path('mask_tol100.png'), 0)  # B&W
        color = (58, 36, 38)
        tolerance = 101
        mask = histonets.color_mask(image, color, tolerance)
        assert np.array_equal(image_mask, mask)

    def test_select_colors(self):
        image = cv2.imread(image_path('poster_kmeans4.png'))
        masked = cv2.imread(image_path('masked_colors.png'))
        colors = (((58, 36, 38), 0), ((172, 99, 76), 0))
        mask = histonets.select_colors(image, colors)
        assert np.array_equal(masked, mask)

    def test_select_colors_as_mask(self):
        image = cv2.imread(image_path('poster_kmeans4.png'))
        masked = cv2.imread(image_path('masked_bw.png'), 0)
        colors = (((58, 36, 38), 0), ((172, 99, 76), 0))
        mask = histonets.select_colors(image, colors, return_mask=True)
        assert np.array_equal(masked, mask)
