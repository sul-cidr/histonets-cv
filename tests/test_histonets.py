#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_histonets
----------------------------------

Tests for `histonets` module.
"""
import os
import unittest

import numpy as np
import cv2
from click.testing import CliRunner

import histonets
from histonets import cli


class TestHistonets(unittest.TestCase):
    def setUp(self):
        self.image = cv2.imread(os.path.join('tests', 'test.png'))

    def test_adjust_high_contrast(self):
        image = self.image
        image_high_contrast = histonets.adjust_contrast(image, 50)
        test_high_contrast = cv2.imread('tests/test_high_contrast.png')
        assert np.array_equal(image_high_contrast, test_high_contrast)

    def test_adjust_low_contrast(self):
        image = self.image
        image_low_contrast = histonets.adjust_contrast(image, -50)
        test_low_contrast = cv2.imread('tests/test_low_contrast.png')
        assert np.array_equal(image_low_contrast, test_low_contrast)

    def test_lower_brightness(self):
        image = self.image
        image_low_brightness = histonets.adjust_brightness(image, -50)
        test_low_brightness = cv2.imread('tests/test_brightness_darken.png')
        assert np.array_equal(image_low_brightness, test_low_brightness)

    def test_higher_brightness(self):
        image = self.image
        image_high_brightness = histonets.adjust_brightness(image, 50)
        test_high_brightness = cv2.imread('tests/test_brightness_lighten.png')
        assert np.array_equal(image_high_brightness, test_high_brightness)


class TestHistonetsCli(unittest.TestCase):
    def setUp(self):
        self.image_url = 'http://httpbin.org/image/jpeg'
        self.image_file = 'file://' + os.path.join('tests', 'test.png')
        self.image_404 = 'file:///not_found.png'
        self.runner = CliRunner()

    def tearDown(self):
        pass

    def test_command_line_interface(self):
        result = self.runner.invoke(cli.main)
        assert result.exit_code == 0
        help_result = self.runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

    def test_download_command_image_file(self):
        result = self.runner.invoke(cli.download, [self.image_file])
        assert 'Error' not in result.output
        assert len(result.output) > 1

    def test_download_command_image_url(self):
        result = self.runner.invoke(cli.download, [self.image_url])
        assert 'Error' not in result.output
        assert len(result.output) > 1

    def test_download_command_image_ot_found(self):
        result = self.runner.invoke(cli.download, [self.image_404])
        assert 'Error' in result.output
