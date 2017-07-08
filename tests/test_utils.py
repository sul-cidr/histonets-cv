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
from imutils import object_detection
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs

from histonets import utils


def fixtures_path(file, relative=False):
    relative_path = os.path.join('tests', 'fixtures', file)
    if not relative:
        return os.path.abspath(relative_path)
    else:
        return relative_path


class TestHistonetsUtils(unittest.TestCase):
    def setUp(self):
        self.image_png = fixtures_path('test.png')
        self.image_file = 'file://' + self.image_png
        self.image_5050_b64 = fixtures_path('test_5050.b64')
        self.image_404 = 'file:///not_found.png'
        self.image_clean = fixtures_path('clean1.png', relative=True)
        self.json = fixtures_path('file.json')
        self.json_gz = fixtures_path('file.json.gz')
        self.json_gz_file = 'file://' + fixtures_path('file.json.gz')
        self.json_content = {"key1": "value1", "key2": 2}

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

    def test_local_decoding(self):
        encoding = locale.getpreferredencoding(False)
        byte = 'Ñoño'.encode(encoding)
        assert utils.local_decode(byte) == byte.decode(encoding)

    def test_parse_pipeline_json(self):
        string = ('[{"action": "brightness", "options": {"value": 150}},'
                  ' {"action": "contrast", "options": {"value": 150}}]')
        obj = [
            {'action': 'brightness', 'options': {'value': 150}},
            {'action': 'contrast', 'options': {'value': 150}}
        ]
        assert utils.parse_pipeline_json(None, None, string) == obj

    def test_parse_pipeline_json_invalid(self):
        string = ('[{"action": "brightness", "options": {"value": 50}},'
                  ' {"actions": "contrast", "options": {"value": 50}}]')
        self.assertRaises(click.BadParameter, utils.parse_pipeline_json,
                          None, None, string)

    def test_parse_pipeline_json_bad_format(self):
        string = ('[*{"action": "brightness", "options": {"value": 50}-}')
        self.assertRaises(click.BadParameter, utils.parse_pipeline_json,
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

    def test_parse_jsons(self):
        string = ['[[[1,2],[5,7],[10,23],[2,3]]]']
        obj = [[[[1, 2], [5, 7], [10, 23], [2, 3]]]]
        assert utils.parse_jsons(None, None, string) == obj

    def test_parse_json_invalid(self):
        string = ['[[[1,2],[5,7],[10,23,[2,3]]]']
        self.assertRaises(click.BadParameter, utils.parse_jsons,
                          None, None, string)

    def test_get_mask_polygons(self):
        output = np.array(
           [[0, 0, 0, 0],
            [0, 255, 255, 255],
            [0, 255, 255, 255],
            [0, 0, 0, 0]], dtype=np.uint8)
        polygons = [[[1, 1], [3, 1], [1, 2], [3, 2]]]
        shape = (4, 4)
        assert np.array_equal(utils.get_mask_polygons(polygons, *shape),
                              output)

    def test_match_template_mask(self):
        image = utils.Image.get_images([self.image_file])[0].image
        template = cv2.imread(fixtures_path('template_m.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        exclude = [
            [[0, 0], [170, 0], [170, 50], [0, 50]],
            [[0, 50], [50, 50], [50, 82], [0, 82]],
            [[120, 50], [170, 50], [170, 82], [120, 82]],
            [[0, 82], [170, 82], [170, 132], [0, 132]],
        ]
        height, width = template.shape[:2]
        mask = ~utils.get_mask_polygons(exclude, height, width)
        correct_match = [[[209, 299], [379, 431]]]
        methods = {
            None: ([[[269, 325], [439, 457]]], 0.14),  # fails at matching
            'canny': (correct_match, 0.45),  # 0.35 for color
            'laplacian': (correct_match, 0.21),
            'sobel': (correct_match, 0.32),
            'scharr': (correct_match, 0.32),
            'prewitt': (correct_match, 0.32),
            'roberts': (correct_match, 0.29),
        }
        for method, (test_matches, threshold) in methods.items():
            results = utils.match_template_mask(image, template, mask, method)
            overlap = 0.15
            index = results >= threshold
            y1, x1 = np.where(index)
            y2, x2 = y1 + height, x1 + width
            coords = np.array([x1, y1, x2, y2], dtype=int).T
            probs = results[index]
            boxes = np.array(
                object_detection.non_max_suppression(coords, probs, overlap)
            )
            matches = boxes.reshape(boxes.shape[0], 2, 2)
            assert np.array_equal(test_matches, matches)

    def test_parse_colors(self):
        colors = ['[1,2,3]', '[123,123,123]']
        obj = [(1, 2, 3), (123, 123, 123)]
        assert utils.parse_colors(None, None, colors) == obj

    def test_parse_colors_malformed(self):
        colors = ['[1,2,3', '[123,123,123]']
        self.assertRaises(click.BadParameter, utils.parse_colors,
                          None, None, colors)

    def test_parse_colors_invalid(self):
        colors = ['[1,2,300]', '[123,123,123]']
        self.assertRaises(click.BadParameter, utils.parse_colors,
                          None, None, colors)

    def test_parse_colors_hex(self):
        colors = ['#abc', '#aabbcc', '[123, 123, 123]']
        obj = [(170, 187, 204), (170, 187, 204), (123, 123, 123)]
        assert utils.parse_colors(None, None, colors) == obj

    def test_parse_colors_hex_malformed(self):
        colors = ['#aabbc', '[123,123,123]']
        self.assertRaises(click.BadParameter, utils.parse_colors,
                          None, None, colors)

    def test_parse_colors_hex_invalid(self):
        colors = ['#abt', '[123,123,123]']
        self.assertRaises(click.BadParameter, utils.parse_colors,
                          None, None, colors)

    def test_convert(self):
        array = np.array([[0, 1, 0], [1, 0, 0]])
        converted = np.array([[0, 255, 0], [255, 0, 0]])
        assert np.array_equal(utils.convert(array), converted)

    def test_output_as_mask(self):
        image = 255 * np.ones((10, 10), np.uint8)
        mask = np.zeros(image.shape, np.uint8)
        masked = cv2.bitwise_and(image, image, mask=mask)
        func = utils.output_as_mask(lambda x: (x, mask))
        assert np.array_equal(masked, func(image))
        assert np.array_equal(mask, func(image, return_mask=True))

    def test_click_choice(self):
        choice = utils.Choice([1, 2, 3])
        assert choice.get_metavar(None) == '[1|2|3]'

    def test_stream(self):
        stream = utils.Stream().convert(self.json)
        content = json.loads(stream)
        assert content == self.json_content

    def test_stream_gz(self):
        stream = utils.Stream().convert(self.json_gz)
        content = json.loads(stream)
        assert content == self.json_content

    def test_stream_gz_using_protocol(self):
        stream = utils.Stream().convert(self.json_gz_file)
        content = json.loads(stream)
        assert content == self.json_content

    def test_stream_bad_scheme(self):
        with self.assertRaises(click.BadParameter):
            utils.Stream().convert('ftp://' + self.json)

    def test_jsonstream(self):
        content = utils.JSONStream().convert(self.json)
        assert content == self.json_content

    def test_jsonstream_gz(self):
        content = utils.JSONStream().convert(self.json_gz)
        assert content == self.json_content

    def test_jsonstream_gz_using_protocol(self):
        content = utils.JSONStream().convert(self.json_gz_file)
        assert content == self.json_content

    def test_jsonstream_raw(self):
        content = utils.JSONStream().convert(json.dumps(self.json_content))
        assert content == self.json_content
