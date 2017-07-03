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
        self.image_template = 'file://' + image_path('template.png')
        self.image_template_h = 'file://' + image_path('template_h.png')
        self.image_template_v = 'file://' + image_path('template_v.png')
        self.image_template_b = 'file://' + image_path('template_b.png')
        self.image_template_m = 'file://' + image_path('template_m.png')
        self.image_posterized = 'file://' + image_path('poster_kmeans4.png')
        self.image_map = 'file://' + image_path('map.png')
        self.image_map_ridges = 'file://' + image_path('map_ridges_invert.png')
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

    def test_command_pipeline_reraise_error(self):
        actions = json.dumps([
            {"action": "posterize", "options":
                {"value": 4, "method": "linear"}}
        ])
        result = self.runner.invoke(cli.pipeline, [actions, self.image_file])
        assert 'Error' in result.output
        assert not isinstance(result.exception, TypeError)
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

    def test_command_clean(self):
        result = self.runner.invoke(cli.clean, [self.image_file])
        assert 'Error' not in result.output
        assert len(result.output.strip()) > 0
        test_clean = encode_base64_image(image_path('clean.png'))
        image = encode_base64_image(self.image_png)
        assert abs(np.ceil(len(result.output.strip()) / 1e5)
                   - np.ceil(len(test_clean) / 1e5)) <= 2
        assert len(test_clean) < len(image)
        assert len(result.output.strip()) < len(image)

    def test_command_enhance(self):
        result_clean = self.runner.invoke(cli.clean, [self.image_file])
        result_enhance = self.runner.invoke(cli.enhance, [self.image_file])
        assert abs(np.ceil(len(result_clean.output) / 1e5)
                   - np.ceil(len(result_enhance.output) / 1e5)) <= 2

    def test_command_match(self):
        result = self.runner.invoke(
            cli.match,
            [self.image_template, '-th', 95, self.image_file]
        )
        assert 'Error' not in result.output
        assert [[[259, 349], [329, 381]]] == json.loads(result.output)

    def test_command_match_default(self):
        result_default = self.runner.invoke(
            cli.match,
            [self.image_template, self.image_file]
        )
        result = self.runner.invoke(
            cli.match,
            [self.image_template, '-th', 80, self.image_file]
        )
        assert 'Error' not in result.output
        assert 'Error' not in result_default.output
        assert result_default.output == result.output

    def test_command_match_invalid(self):
        result = self.runner.invoke(
            cli.match,
            [self.image_template, '-th', 95, '-th', 80, self.image_file]
        )
        assert 'Error' in result.output

    def test_command_match_flip_horizontally(self):
        result = self.runner.invoke(
            cli.match,
            [self.image_template, '-th', 95, self.image_file]
        )
        assert 'Error' not in result.output
        result_h = self.runner.invoke(
            cli.match,
            [self.image_template_h, '-th', 95, '-f', 'h', self.image_file]
        )
        assert 'Error' not in result_h.output
        assert result.output == result_h.output

    def test_command_match_flip_vertically(self):
        result = self.runner.invoke(
            cli.match,
            [self.image_template, '-th', 95, self.image_file]
        )
        assert 'Error' not in result.output
        result_v = self.runner.invoke(
            cli.match,
            [self.image_template_v, '-th', 95, '-f', 'v', self.image_file]
        )
        assert 'Error' not in result_v.output
        assert result.output == result_v.output

    def test_command_match_flip_both(self):
        result = self.runner.invoke(
            cli.match,
            [self.image_template, '-th', 95, self.image_file]
        )
        assert 'Error' not in result.output
        result_b = self.runner.invoke(
            cli.match,
            [self.image_template_b, '-th', 95, '-f', 'b', self.image_file]
        )
        assert 'Error' not in result_b.output
        assert result.output == result_b.output

    def test_command_match_mask(self):
        exclude = [
            [[0, 0], [170, 0], [170, 50], [0, 50]],
            [[0, 50], [50, 50], [50, 82], [0, 82]],
            [[120, 50], [170, 50], [170, 82], [120, 82]],
            [[0, 82], [170, 82], [170, 132], [0, 132]],
        ]
        result = self.runner.invoke(
            cli.match,
            [self.image_template_m, '-th', 45,
             '-e', json.dumps(exclude), self.image_file]
        )
        test_matches = [[[209, 299], [379, 431]]]
        assert 'Error' not in result.output
        assert test_matches == json.loads(result.output)

    def test_command_match_mask_invalid(self):
        result = self.runner.invoke(
            cli.match,
            [self.image_template, '-th', 95,
             '-e', '[[[1,2],[5,7],[10,23,[2,3]]]', self.image_file]
        )
        assert 'Error' in result.output
        assert 'Polygon' in result.output

    def test_command_pipeline_match(self):
        exclude = [
            [[0, 0], [170, 0], [170, 50], [0, 50]],
            [[0, 50], [50, 50], [50, 82], [0, 82]],
            [[120, 50], [170, 50], [170, 82], [120, 82]],
            [[0, 82], [170, 82], [170, 132], [0, 132]],
        ]
        actions = json.dumps([
            {"action": "match", "options": {
                "templates": [self.image_template_m],
                "threshold": [45],
                "flip": ['both'],
                "exclude_regions": [exclude],
            }},
        ])
        result = self.runner.invoke(cli.pipeline, [actions, self.image_file])
        test_matches = [[[209, 299], [379, 431]]]
        assert 'Error' not in result.output
        assert test_matches == json.loads(result.output)

    def test_command_select_colors(self):
        result = self.runner.invoke(
            cli.select,
            [json.dumps((58, 36, 38)), '-t', 0,
             json.dumps((172, 99, 76)), '-t', 0,
             self.image_posterized]
        )
        masked = encode_base64_image(image_path('masked_colors.png'))
        assert masked == result.output.strip()

    def test_command_select_colors_as_mask(self):
        result = self.runner.invoke(
            cli.select,
            [json.dumps((58, 36, 38)), '-t', 0,
             json.dumps((172, 99, 76)), '-t', 0,
             '--mask',
             self.image_posterized]
        )
        masked = encode_base64_image(image_path('masked_bw.png'))
        assert masked == result.output.strip()

    def test_command_ridges(self):
        result = self.runner.invoke(
            cli.ridges,
            ['-w', 6, '-th', 160, '-d', 3,
             self.image_map]
        )
        masked = encode_base64_image(image_path('map_noridge.png'))
        assert masked == result.output.strip()

    def test_command_ridges_as_mask(self):
        result = self.runner.invoke(
            cli.ridges,
            ['-w', 6, '-th', 160, '-d', 3, '-m',
             self.image_map]
        )
        mask = encode_base64_image(image_path('map_ridge.png'))
        assert mask == result.output.strip()

    def test_command_blobs(self):
        result = self.runner.invoke(
            cli.blobs,
            ['-min', 0, '-max', 100,
             self.image_map_ridges],
        )
        masked = encode_base64_image(image_path('map_noblobs8.png'))
        assert masked == result.output.strip()

    def test_command_blobs_4connected(self):
        result = self.runner.invoke(
            cli.blobs,
            ['-min', 0, '-max', 100, '-c', 4,
             self.image_map_ridges],
        )
        masked = encode_base64_image(image_path('map_noblobs4.png'))
        assert masked == result.output.strip()

    def test_command_blobs_8connected(self):
        result = self.runner.invoke(
            cli.blobs,
            ['-min', 0, '-max', 100, '-c', 8,
             self.image_map_ridges],
        )
        masked = encode_base64_image(image_path('map_noblobs8.png'))
        assert masked == result.output.strip()

    def test_command_blobs_antialiased(self):
        result = self.runner.invoke(
            cli.blobs,
            ['-min', 0, '-max', 100, '-c', 16,
             self.image_map_ridges],
        )
        masked = encode_base64_image(image_path('map_noblobs_antialiased.png'))
        assert masked == result.output.strip()

    def test_binarize_default(self):
        result = self.runner.invoke(
            cli.binarize,
            [self.image_map],
        )
        binarized = encode_base64_image(image_path('map_bw.png'))
        assert binarized == result.output.strip()

    def test_binarize_li(self):
        result = self.runner.invoke(
            cli.binarize,
            ['-m', 'li',
             self.image_map],
        )
        binarized = encode_base64_image(image_path('map_bw.png'))
        assert binarized == result.output.strip()

    def test_binarize_otsu(self):
        result = self.runner.invoke(
            cli.binarize,
            ['-m', 'otsu',
             self.image_map],
        )
        binarized = encode_base64_image(image_path('map_otsu.png'))
        assert binarized == result.output.strip()

    def test_skeletonize(self):
        result = self.runner.invoke(
            cli.skeletonize,
            [self.image_map],
        )
        skeleton = encode_base64_image(image_path('map_sk_combined_d13.png'))
        assert skeleton == result.output.strip()

    def test_skeletonize_default(self):
        result = self.runner.invoke(
            cli.skeletonize,
            ['-m', 'combined', '-b', 'li', '-d', 13,
             self.image_map],
        )
        skeleton = encode_base64_image(image_path('map_sk_combined_d13.png'))
        assert skeleton == result.output.strip()

    def test_skeletonize_no_dilation_thin(self):
        result = self.runner.invoke(
            cli.skeletonize,
            ['-m', 'thin', '-d', 0,
             self.image_map],
        )
        skeleton = encode_base64_image(image_path('map_sk_thin_d0.png'))
        assert skeleton == result.output.strip()
