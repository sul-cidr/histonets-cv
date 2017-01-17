#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_utils
----------------------------------

Tests for `histonets.utils` module.
"""
import io
import json
import os
import locale
import subprocess
import unittest

import click
import cv2
import noteshrink
import numpy as np
from click.testing import CliRunner
from collections import namedtuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs

from histonets import utils


def image_path(img):
    return os.path.join('tests', 'images', img)


class TestHistonetsUtils(unittest.TestCase):
    def setUp(self):
        self.image_png = image_path('test.png')
        self.image_file = 'file://' + self.image_png
        self.image_5050_b64 = image_path('test_5050.b64')
        self.image_404 = 'file:///not_found.png'
        self.image_clean = 'file://' + image_path('clean1.png')

    def test_get_images(self):
        images = utils.Image.get_images([self.image_file, self.image_file])
        for image in images:
            assert isinstance(image, utils.Image)

    def test_get_images_callback(self):
        images = utils.Image.get_images([self.image_file, self.image_file])
        images_callback = utils.get_images(None, None,
                                           [self.image_file, self.image_file])
        for image, images_callback in zip(*[images, images_callback]):
            assert isinstance(images_callback, utils.Image)
            assert np.array_equal(image.image, images_callback.image)

    def test_get_images_callback_invalid(self):
        with self.assertRaises(click.BadParameter):
            utils.get_images(None, None, [self.image_file, self.image_404])

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

    def test_get_color_histogram(self):
        image = utils.Image.get_images([self.image_file])[0]
        colors = utils.get_color_histogram(image)
        assert len(colors) == 59823

    def test_kmeans(self):
        n_clusters = 5
        X, y = make_blobs(n_samples=1000, centers=n_clusters, random_state=0)
        centers, labels = utils.kmeans(X, n_clusters)
        clf = MiniBatchKMeans(n_clusters=n_clusters)
        assert len(labels) == len(clf.fit_predict(X))
        assert len(centers) == len(clf.cluster_centers_)

    def test_get_palette_min_values(self):
        image = utils.Image.get_images([self.image_clean])[0].image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        options = namedtuple(
            'options',
            ['quiet', 'sample_fraction', 'value_threshold', 'sat_threshold']
        )(
            quiet=True,
            sample_fraction=.01,
            value_threshold=.01,
            sat_threshold=.01,
        )
        samples = noteshrink.sample_pixels(rgb_image, options)
        palette = utils.get_palette(samples, 2, background_value=1,
                                    background_saturation=1)
        test_palette = np.array([[254, 122, 94], [193, 86, 64]])
        assert palette.shape == test_palette.shape
        # background colors must coincide
        assert np.array_equal(palette[0], test_palette[0])

    def test_get_palette_max_values(self):
        image = utils.Image.get_images([self.image_clean])[0].image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        options = namedtuple(
            'options',
            ['quiet', 'sample_fraction', 'value_threshold', 'sat_threshold']
        )(
            quiet=True,
            sample_fraction=1,
            value_threshold=1,
            sat_threshold=1,
        )
        samples = noteshrink.sample_pixels(rgb_image, options)
        palette = utils.get_palette(samples, 128, background_value=100,
                                    background_saturation=100)
        background_color = np.array([254, 122, 94])
        assert palette.shape == (128, 3)
        # background colors must coincide
        assert np.array_equal(palette[0], background_color)

    def test_pair_options_to_argument_args(self):
        args = ['im', 't1', '-o', '1', 't2', 't3', '-o', '3']

        @click.command()
        @click.argument('img')
        @click.argument('arg', nargs=-1, required=True)
        @click.option('-o', '--option', multiple=True)
        @utils.pair_options_to_argument(
            'arg', {'option': 0}, args=args, args_slice=(1, None)
        )
        def command(img, arg, option):
            click.echo(json.dumps((arg, option)))

        runner = CliRunner()
        output = runner.invoke(command, args).output
        assert 'Error' not in output
        assert [["t1", "t2", "t3"], ["1", 0, "3"]] == json.loads(output)

    def test_two_paired_options_to_argument_args(self):
        args = ['im', 't1', '-o', '1', 't2', '-a', '3', 't3']

        @click.command()
        @click.argument('img')
        @click.argument('arg', nargs=-1, required=True)
        @click.option('-o', '--option', multiple=True)
        @click.option('-a', '--another', multiple=True)
        @utils.pair_options_to_argument(
            'arg', {'option': 0, 'another': 1}, args=args, args_slice=(1, None)
        )
        def command(img, arg, option, another):
            click.echo(json.dumps((arg, option, another)))

        runner = CliRunner()
        output = runner.invoke(command, args).output
        assert 'Error' not in output
        assert ([["t1", "t2", "t3"], ["1", 0, 0], [1, "3", 1]]
                    == json.loads(output))

    def test_pair_options_to_argument_args_default(self):
        args = ['im', 't1', 't2', 't3']

        @click.command()
        @click.argument('img')
        @click.argument('arg', nargs=-1, required=True)
        @click.option('-o', '--option', multiple=True)
        @utils.pair_options_to_argument(
            'arg', {'option': 0}, args=args, args_slice=(1, None)
        )
        def command(img, arg, option):
            click.echo(json.dumps((arg, option)))

        runner = CliRunner()
        output = runner.invoke(command, args).output
        assert 'Error' not in output
        assert [["t1", "t2", "t3"], [0, 0, 0]] == json.loads(output)

    def test_pair_options_to_argument(self):
        code = """
import json
import click
from histonets import utils
@click.group()
def main():
    pass
@main.command()
@click.argument('img')
@click.argument('arg', nargs=-1, required=True)
@click.option('-o', '--option', multiple=True)
@utils.pair_options_to_argument('arg', {'option': 0}, args_slice=(2, None))
def command(img, arg, option):
    click.echo(json.dumps((arg, option)))
main()
        """
        cmd = ("echo \"{}\" "
               "| python - command im t1 -o 1 t2 t3 -o 3".format(code))
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        output = ps.communicate()[0].decode()
        assert 'Error' not in output
        assert [["t1", "t2", "t3"], ["1", 0, "3"]] == json.loads(output)
