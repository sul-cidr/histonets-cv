#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_utils
----------------------------------

Tests for `histonets.utils` module.
"""
import io
import os
import locale
import subprocess
import unittest

import numpy as np
import click

from histonets import utils


def image_path(img):
    return os.path.join('tests', 'images', img)


class TestHistonetsUtils(unittest.TestCase):
    def setUp(self):
        self.image_png = image_path('test.png')
        self.image_file = 'file://' + self.image_png
        self.image_5050_b64 = image_path('test_5050.b64')
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
               " | histonets brightness 150"
               " | histonets contrast 150".format(
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
        string = ('[{"action": "brightness", "options": {"value": 150}},'
                  ' {"action": "contrast", "options": {"value": 150}}]')
        obj = [
            {'action': 'brightness', 'options': {'value': 150}},
            {'action': 'contrast', 'options': {'value': 150}}
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
        func = utils.image_as_array(lambda x: x)
        assert np.array_equal(image.image, func(image))
        assert np.array_equal(image.image, func(image.image))

    def test_get_palette(self):
        image = utils.Image.get_images([self.image_file])[0]
        colors = utils.get_palette(image)
        assert len(colors) == 59823
