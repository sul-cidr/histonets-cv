#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_histonets
----------------------------------

Tests for `histonets` module.
"""
import io
import json
import os
import locale
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
import click
from click.testing import CliRunner

import histonets
from histonets import cli, utils


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

    def test_contrast_value_parsing(self):
        image = self.image
        assert np.array_equal(
            histonets.adjust_contrast(image, -150),
            histonets.adjust_contrast(image, -100)
        )
        assert np.array_equal(
            histonets.adjust_contrast(image, 150),
            histonets.adjust_contrast(image, 100)
        )

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

    def test_brightness_value_parsing(self):
        image = self.image
        assert np.array_equal(
            histonets.adjust_brightness(image, -150),
            histonets.adjust_brightness(image, -100)
        )
        assert np.array_equal(
            histonets.adjust_brightness(image, 150),
            histonets.adjust_brightness(image, 100)
        )

    def test_smooth_image(self):
        image = self.image
        smooth_image = histonets.smooth_image(image, 50)
        test_smooth_image = cv2.imread('tests/smooth50.png')
        assert np.array_equal(smooth_image, test_smooth_image)

    def test_smooth_image_value_parsing(self):
        image = self.image
        test_smooth100_image = cv2.imread('tests/smooth100.png')
        test_smooth0_image = cv2.imread('tests/smooth0.png')
        assert np.array_equal(
            histonets.smooth_image(image, 150),
            test_smooth100_image
            )
        assert np.array_equal(
            histonets.smooth_image(image, -50),
            test_smooth0_image
            )


class TestHistonetsCli(unittest.TestCase):
    def setUp(self):
        self.image_url = 'http://httpbin.org/image/jpeg'
        self.image_file = 'file://' + os.path.join('tests', 'test.png')
        self.image_404 = 'file:///not_found.png'
        self.image_jpg = os.path.join('tests', 'test.jpg')
        self.image_png = os.path.join('tests', 'test.png')
        self.image_b64 = os.path.join('tests', 'test.b64')
        self.image_5050_b64 = os.path.join('tests', 'test_5050.b64')
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

    def test_io_handler_to_file_as_png(self):
        result = self.runner.invoke(cli.download,
            [self.image_file, '--output', self.tmp_png]
        )
        assert 'Error' not in result.output
        assert len(result.output) == 0
        image_png = cv2.imread(self.image_png)
        tmp_png = cv2.imread(self.tmp_png)
        assert np.array_equal(image_png, tmp_png)

    def test_io_handler_to_file_as_jpg(self):
        result = self.runner.invoke(cli.download,
            [self.image_file, '--output', self.tmp_jpg]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) == 0
        image_jpg = cv2.imread(self.image_jpg)
        tmp_jpg = cv2.imread(self.tmp_jpg)
        assert np.array_equal(image_jpg, tmp_jpg)

    def test_io_handler_to_file_as_png(self):
        result = self.runner.invoke(cli.download,
            [self.image_file, '--output', self.tmp_png]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) == 0
        image_png = cv2.imread(self.image_png)
        tmp_png = cv2.imread(self.tmp_png)
        assert np.array_equal(image_png, tmp_png)

    def test_io_handler_to_file_and_convert_to_tiff(self):
        result = self.runner.invoke(cli.download,
            [self.image_file, '--output', self.tmp_tiff]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) == 0
        image_png = cv2.imread(self.image_png)
        tmp_tiff = cv2.imread(self.tmp_tiff)
        assert np.array_equal(image_png, tmp_tiff)

    def test_io_handler_to_file_with_no_format(self):
        result = self.runner.invoke(cli.download,
            [self.image_file, '--output', self.tmp_no_format]
        )
        assert 'Error' not in result.output
        assert len(result.output.strip()) == 0
        image_png = cv2.imread(self.image_png)
        tmp_no_format = cv2.imread(self.tmp_no_format)
        assert np.array_equal(image_png, tmp_no_format)

    def test_io_handler_to_file_with_invalid_format(self):
        result = self.runner.invoke(cli.download,
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
        result = self.runner.invoke(cli.contrast, ['150', self.image_file])
        assert 'Invalid value for "value"' in result.output

    def test_brightness_invalid_value(self):
        result = self.runner.invoke(cli.brightness, ['150', self.image_file])
        assert 'Invalid value for "value"' in result.output

    def test_smooth_invalid_value(self):
        result = self.runner.invoke(cli.smooth, ['101', self.image_file])
        assert 'Invalid value for "value"' in result.output

    def test_command_pipeline(self):
        actions = json.dumps([
            {'action': 'brightness', 'options': {'value': 50}},
            {'action': 'contrast', 'options': {'value': 50}}
        ])
        result = self.runner.invoke(cli.pipeline, [actions, self.image_file])
        assert 'Error' not in result.output
        assert len(result.output.strip()) > 0
        with io.open(self.image_5050_b64) as image_b64:
            assert result.output == image_b64.read()

    def test_command_pipeline_invalid(self):
        actions = json.dumps([
            {'action': 'command not found', 'options': {'value': 50}},
        ])
        result = self.runner.invoke(cli.pipeline, [actions, self.image_file])
        assert 'Error' in result.output
        assert len(result.output.strip()) > 0


class TestHistonetsUtils(unittest.TestCase):
    def setUp(self):
        self.image_png = os.path.join('tests', 'test.png')
        self.image_file = 'file://' + self.image_png
        self.image_5050_b64 = os.path.join('tests', 'test_5050.b64')
        self.image_404 = 'file:///not_found.png'

    def test_get_images(self):
        images = utils.Image.get_images([self.image_file, self.image_file])
        for image in images:
            assert isinstance(image, utils.Image)

    def test_get_images_class(self):
        images = utils.Image.get_images([self.image_file, self.image_file])
        assert images == utils.Image.get_images(images)

    def test_images_class(self):
        image = utils.Image.get_images([self.image_file])[0]
        assert np.array_equal(image.image, utils.Image(image=image).image)
        assert np.array_equal(image.format, utils.Image(image=image).format)
        assert np.array_equal(image.image,
                              utils.Image(image=image.image).image)

    def test_get_images_invalid(self):
        self.assertRaises(
            click.BadParameter,
            utils.Image.get_images,
            [self.image_file, self.image_404]
        )

    def test_get_images_stdin(self):
        cmd = ("python tests/encode_image.py -i {}"
               " | histonets brightness 50"
               " | histonets contrast 50".format(
                    self.image_png
                ))
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        output = ps.communicate()[0].decode()
        with io.open(self.image_5050_b64) as image_b64:
            assert output == image_b64.read()

    def test_local_encoding(self):
        string = 'Ñoño'
        encoding = locale.getpreferredencoding(False)
        assert utils.local_encode(string) == string.encode(encoding)

    def test_parse_json(self):
        string = ('[{"action": "brightness", "options": {"value": 50}},'
                  ' {"action": "contrast", "options": {"value": 50}}]')
        obj = [
            {'action': 'brightness', 'options': {'value': 50}},
            {'action': 'contrast', 'options': {'value': 50}}
        ]
        assert utils.parse_json(None, None, string) == obj

    def test_parse_json_invalid(self):
        string = ('[{"action": "brightness", "options": {"value": 50}},'
                  ' {"actions": "contrast", "options": {"value": 50}}]')
        self.assertRaises(click.BadParameter, utils.parse_json,
                          None, None, string)

    def test_parse_json_bad_format(self):
        string = ('[*{"action": "brightness", "options": {"value": 50}-}')
        self.assertRaises(click.BadParameter, utils.parse_json,
                          None, None, string)

    def test_image_as_array(self):
        image = utils.Image.get_images([self.image_file])[0]
        func = lambda x: x
        func = utils.image_as_array(func)
        assert np.array_equal(image.image, func(image))
        assert np.array_equal(image.image, func(image.image))
