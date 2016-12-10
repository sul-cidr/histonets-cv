#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_cli
----------------------------------

Tests for `histonets.cli` module.
"""
import base64
import io
import json
import os
import tempfile
import unittest

import cv2
import numpy as np
from click.testing import CliRunner

from histonets import cli


def image_path(img):
    return os.path.join('tests', 'images', img)


def encode_base64_image(img_path):
    with open(img_path, 'rb') as image:
        return base64.b64encode(image.read()).decode()


class TestHistonetsCli(unittest.TestCase):
    def setUp(self):
        self.image_url = 'http://httpbin.org/image/jpeg'
        self.image_file = 'file://' + image_path('test.png')
        self.image_404 = 'file:///not_found.png'
        self.image_jpg = image_path('test.jpg')
        self.image_png = image_path('test.png')
        self.image_b64 = image_path('test.b64')
        self.image_5050_b64 = image_path('test_5050.b64')
        self.tmp_jpg = os.path.join(tempfile.gettempdir(), 'test.jpg')
        self.tmp_png = os.path.join(tempfile.gettempdir(), 'test.png')
        self.tmp_tiff = os.path.join(tempfile.gettempdir(), 'test.tiff')
        self.tmp_no_format = os.path.join(tempfile.gettempdir(), 'test')
        self.tmp_invalid_format = os.path.join(tempfile.gettempdir(), 'test.a')
        self.runner = CliRunner()

    def tearDown(self):
        pass

    def test_command_line_interface(self):
        result = self.runner.invoke(cli.main)
        assert result.exit_code == 0
        help_result = self.runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help' in help_result.output
        assert 'Show this message and exit.' in help_result.output
        assert '--version' in help_result.output
        assert 'Show the version and exit.' in help_result.output
        assert 'Download IMAGE.' in result.output

    def test_rst_option(self):
        result = self.runner.invoke(cli.main)
        assert result.exit_code == 0
        help_result = self.runner.invoke(cli.main, ['--rst'])
        assert help_result.exit_code == 0
        assert '~' in help_result.output
        assert 'Commands' in help_result.output
        assert 'Options' in help_result.output

    def test_download_command_image_file(self):
        result = self.runner.invoke(cli.download, [self.image_file])
        assert 'Error' not in result.output
        assert len(result.output.strip()) > 1

    def test_download_command_image_url(self):
        result = self.runner.invoke(cli.download, [self.image_url])
        assert 'Error' not in result.output
        assert len(result.output.strip()) > 1

    def test_download_command_image_not_found(self):
        result = self.runner.invoke(cli.download, [self.image_404])
        assert 'Error' in result.output

    def test_download_command_help(self):
        result = self.runner.invoke(cli.download, ['--help'])
        assert 'Download IMAGE.' in result.output

    def test_io_handler_to_file_as_jpg(self):
        result = self.runner.invoke(
            cli.download,
            [self.image_file, '--output', self.tmp_jpg]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) == 0
        image_jpg = cv2.imread(self.image_jpg)
        tmp_jpg = cv2.imread(self.tmp_jpg)
        assert np.array_equal(image_jpg, tmp_jpg)

    def test_io_handler_to_file_as_png(self):
        result = self.runner.invoke(
            cli.download,
            [self.image_file, '--output', self.tmp_png]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) == 0
        image_png = cv2.imread(self.image_png)
        tmp_png = cv2.imread(self.tmp_png)
        assert np.array_equal(image_png, tmp_png)

    def test_io_handler_to_file_and_convert_to_tiff(self):
        result = self.runner.invoke(
            cli.download,
            [self.image_file, '--output', self.tmp_tiff]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) == 0
        image_png = cv2.imread(self.image_png)
        tmp_tiff = cv2.imread(self.tmp_tiff)
        assert np.array_equal(image_png, tmp_tiff)

    def test_io_handler_to_file_with_no_format(self):
        result = self.runner.invoke(
            cli.download,
            [self.image_file, '--output', self.tmp_no_format]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) == 0
        image_png = cv2.imread(self.image_png)
        tmp_no_format = cv2.imread(self.tmp_no_format)
        assert np.array_equal(image_png, tmp_no_format)

    def test_io_handler_to_file_with_invalid_format(self):
        result = self.runner.invoke(
            cli.download,
            [self.image_file, '--output', self.tmp_invalid_format]
        )
        assert 'Error' in result.output
        assert len(result.output.strip()) > 0

    def test_io_handler_to_stdout(self):
        result = self.runner.invoke(cli.download, [self.image_file])
        assert 'Error' not in result.output
        assert len(result.output.strip()) > 0
        with io.open(self.image_b64) as image_b64:
            assert result.output == image_b64.read()

    def test_contrast_invalid_value(self):
        result = self.runner.invoke(cli.contrast, ['250', self.image_file])
        assert 'Invalid value for "value"' in result.output

    def test_contrast_integer(self):
        test_contrast_image = encode_base64_image(
            image_path('test_low_contrast.png')
        )
        result = self.runner.invoke(cli.contrast, ['50', self.image_file])
        assert test_contrast_image == result.output.rstrip()

    def test_brightness_invalid_value(self):
        result = self.runner.invoke(cli.brightness, ['250', self.image_file])
        assert 'Invalid value for "value"' in result.output

    def test_brightness_integer(self):
        test_brightness_image = encode_base64_image(
            image_path('test_brightness_darken.png')
        )
        result = self.runner.invoke(cli.brightness, ['50', self.image_file])
        assert test_brightness_image == result.output.rstrip()

    def test_smooth_invalid_value(self):
        result = self.runner.invoke(cli.smooth, ['101', self.image_file])
        assert 'Invalid value for "value"' in result.output

    def test_smooth_integer(self):
        test_smooth_image = encode_base64_image(
            image_path('smooth50.png')
        )
        result = self.runner.invoke(cli.smooth, ['50', self.image_file])
        assert test_smooth_image == result.output.rstrip()

    def test_histogram_equalization_invalid_value(self):
        result = self.runner.invoke(cli.equalize, ['150', self.image_file])
        assert 'Invalid value for "value"' in result.output

    def test_histogram_equalization_integer(self):
        test_hist_image = encode_base64_image(
            image_path('test_hist_eq5.png')
        )
        result = self.runner.invoke(cli.equalize, ['50', self.image_file])
        assert test_hist_image == result.output.rstrip()

    def test_denoise_invalid_value(self):
        result = self.runner.invoke(cli.denoise, ['110', self.image_file])
        assert 'Invalid value for "value"' in result.output

    def test_denoise_integer(self):
        test_denoise_image = encode_base64_image(
            image_path('denoised10.png')
        )
        result = self.runner.invoke(cli.denoise, ['10', self.image_file])
        assert test_denoise_image == result.output.rstrip()

    def test_command_pipeline(self):
        actions = json.dumps([
            {'action': 'brightness', 'options': {'value': 150}},
            {'action': 'contrast', 'options': {'value': 150}}
        ])
        result = self.runner.invoke(cli.pipeline, [actions, self.image_file])
        assert 'Error' not in result.output
        assert len(result.output.strip()) > 0
        with io.open(self.image_5050_b64) as image_b64:
            assert result.output == image_b64.read()

    def test_command_pipeline_all_actions(self):
        actions = json.dumps([
            {"action": "denoise", "options": {"value": 9}},
            {"action": "equalize", "options": {"value": 10}},
            {"action": "brightness", "options": {"value": 122}},
            {"action": "contrast", "options": {"value": 122}},
            {"action": "smooth", "options": {"value": 12}},
            {"action": "posterize", "options":
                {"colors": 4, "method": "linear"}}
        ])
        result = self.runner.invoke(cli.pipeline, [actions, self.image_file])
        assert 'Error' not in result.output
        test_pipeline_full = encode_base64_image(
            image_path('test_full_pipeline.png')
        )
        assert test_pipeline_full == result.output.strip()

    def test_command_pipeline_invalid(self):
        actions = json.dumps([
            {'action': 'command not found', 'options': {'value': 50}},
        ])
        result = self.runner.invoke(cli.pipeline, [actions, self.image_file])
        assert 'Error' in result.output
        assert len(result.output.strip()) > 0

    def test_command_posterize_linear(self):
        result = self.runner.invoke(
            cli.posterize,
            ['4', '-m', 'linear', self.image_file]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) > 0
        test_posterize_image = encode_base64_image(
            image_path('poster_linear4.png')
            )
        assert test_posterize_image == result.output.strip()

    def test_command_posterize_default_method(self):
        result = self.runner.invoke(
            cli.posterize,
            ['4', self.image_file]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) > 0
        test_posterize_image = encode_base64_image(
            image_path('poster_linear4.png')
            )
        assert test_posterize_image != result.output.strip()
