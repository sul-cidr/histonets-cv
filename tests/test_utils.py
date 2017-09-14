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
import networkx as nx
import noteshrink
import numpy as np
from click.testing import CliRunner
from collections import namedtuple
from imutils import object_detection
from networkx.readwrite import json_graph as nx_json_graph
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs

from histonets import utils


def fixtures_path(file, relative=False):
    relative_path = os.path.join('tests', 'fixtures', file)
    if not relative:
        return os.path.abspath(relative_path)
    else:
        return relative_path


def edgeset(graph):
    # We remove property id since it is not consistent in GEXF format
    return set([
        tuple(
            sorted([u, v])
            + sorted((k, v) for k, v in props.items() if k != 'id'))
        for u, v, props in graph.edges(data=True)
    ])


def nodeset(graph):
    return sorted(graph.nodes(data=True))


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
        test_palette = np.array([[255, 123, 92], [193, 86, 64]])
        assert palette.shape <= test_palette.shape
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
        background_color = np.array([255, 123, 92])
        assert palette.shape <= (128, 3)
        # background colors must coincide
        assert np.array_equal(palette[0], background_color)

    def test_get_palette_kmeans(self):
        image = utils.Image.get_images([self.image_png])[0].image
        image_pixels = image.reshape((-1, 3))
        assert (len(utils.get_palette(image_pixels, 4))
                == len(utils.kmeans(image_pixels, 4)[0]))

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
        colors = ['[0,1,2]', '[123,123,123]']
        obj = [(0, 1, 2), (123, 123, 123)]
        assert utils.parse_colors(None, None, colors) == obj

    def test_parse_colors_malformed(self):
        colors = ['[1,2,3', '[123,123,123]']
        self.assertRaises(click.BadParameter, utils.parse_colors,
                          None, None, colors)

    def test_parse_colors_invalid(self):
        colors = ['[1,2,300]', '[123,123,123]']
        self.assertRaises(click.BadParameter, utils.parse_colors,
                          None, None, colors)

    def test_parse_colors_hex_list(self):
        colors = ['#abc', '#aabbcc', [123, 123, 123]]
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
        stream = utils.Stream().convert(value=self.json)
        content = json.loads(stream)
        assert content == self.json_content

    def test_stream_gz(self):
        stream = utils.Stream().convert(value=self.json_gz)
        content = json.loads(stream)
        assert content == self.json_content

    def test_stream_gz_using_protocol(self):
        stream = utils.Stream().convert(value=self.json_gz_file)
        content = json.loads(stream)
        assert content == self.json_content

    def test_stream_bad_scheme(self):
        with self.assertRaisesRegex(click.BadParameter, 'Scheme'):
            utils.Stream().convert(value='ftp://' + self.json)

    def test_stream_file_not_found(self):
        with self.assertRaisesRegex(click.BadParameter, 'File'):
            utils.Stream().convert(value='notfound')

    def test_jsonstream(self):
        content = utils.JSONStream().convert(value=self.json)
        assert content == self.json_content

    def test_jsonstream_gz(self):
        content = utils.JSONStream().convert(value=self.json_gz)
        assert content == self.json_content

    def test_jsonstream_gz_using_protocol(self):
        content = utils.JSONStream().convert(value=self.json_gz_file)
        assert content == self.json_content

    def test_jsonstream_raw(self):
        content = utils.JSONStream().convert(
            value=json.dumps(self.json_content))
        assert content == self.json_content

    def test_parse_histogram(self):
        histogram = '{"#abc": 10, "#aabbdd": "100", "[123, 123, 123]": "1"}'
        histogram_dict = {
            (123, 123, 123): 1,
            (170, 187, 221): 100,
            (170, 187, 204): 10
        }
        assert utils.parse_histogram(histogram) == histogram_dict

    def test_parse_histogram_invalid(self):
        histogram = '{"#abc": 10, "#aabbdd": "1e3", "[123, 123, 123]": "1"}'
        self.assertRaises(click.BadParameter, utils.parse_histogram,
                          histogram)

    def test_parse_histogram_dict(self):
        histogram = {
            (123, 123, 123): 1,
            '#aabbdd': 100,
            '#abc': 10
        }
        histogram_dict = {
            (123, 123, 123): 1,
            (170, 187, 221): 100,
            (170, 187, 204): 10
        }
        assert utils.parse_histogram(histogram) == histogram_dict

    def test_sample_histogram(self):
        histogram = dict(zip(*[
            [tuple(c) for c in np.random.randint(255, size=(5, 3))],
            np.random.randint(100, size=5)]
        ))
        sampled = utils.sample_histogram(histogram)
        assert sampled.shape <= (100 * 5 * 0.05, 3)

    def test_sample_histogram_with_lower_sample_fraction(self):
        histogram = dict(zip(*[
            [tuple(c) for c in np.random.randint(255, size=(5, 3))],
            np.random.randint(100, size=5)]
        ))
        sampled = utils.sample_histogram(histogram, sample_fraction=0.01)
        assert sampled.shape <= (100 * 5 * 0.01, 3)

    def test_parse_palette(self):
        palette_str = '["#abc", "#aabbdd", "[123, 123, 123]", [1, 2, 3]]'
        palette = [
            [170, 187, 204],
            [170, 187, 221],
            [123, 123, 123],
            [1, 2, 3],
        ]
        assert (utils.parse_palette(None, None, palette_str).tolist()
                == palette)

    def test_parse_palette_invalid(self):
        self.assertRaises(click.BadParameter, utils.parse_palette,
                          None, None, '#abc')

    def test_parse_palette_none(self):
        assert utils.parse_palette(None, None, None) is None

    def test_get_quantize_method(self):
        assert utils.get_quantize_method('median') == 0
        assert utils.get_quantize_method('octree') == 2
        assert utils.get_quantize_method('max') == 1

    def test_unique(self):
        assert (utils.unique(['b', 'b', 'b', 'a', 'a', 'c', 'c']).tolist()
                == ['b', 'a', 'c'])

    def test_get_inner_paths(self):
        grid = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 1, 8, 8, 8, 0, 0],
            [0, 1, 8, 8, 8, 1, 0],
            [0, 0, 8, 8, 8, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=bool)
        regions = [((2, 2), (4, 4))]
        output = 255 * np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        assert np.array_equal(utils.get_inner_paths(grid, regions).toarray(),
                              output)

    def test_grid_to_adjacency_matrix(self):
        grid = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
        ], dtype=bool)
        matrix = np.array([
            [1, 1, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ], dtype=bool)
        assert np.array_equal(utils.grid_to_adjacency_matrix(grid).toarray(),
                              matrix)

    def test_grid_to_adjacency_matrix_4_connected(self):
        grid = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
        ], dtype=bool)
        matrix = np.array([
            [1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 1]
        ], dtype=bool)
        assert np.array_equal(
            utils.grid_to_adjacency_matrix(grid, neighborhood=4).toarray(),
            matrix
        )

    def test_shortest_paths(self):
        grid = np.array([
            # 0  1  2
            [1, 1, 1],  # 0
            [1, 0, 1],  # 1
            [0, 0, 1],  # 2
        ], dtype=bool)
        lookfor = ((0, 1), (2, 0)), ((2, 0), (2, 2)), ((0, 1), (2, 2))
        paths = [
            [(0, 1), (1, 0), (2, 0)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 1), (1, 0), (2, 1), (2, 2)],
        ]
        assert np.array_equal(paths, utils.get_shortest_paths(grid, lookfor))

    def test_argfirst2D(self):
        assert utils.argfirst2D(np.array([[1, 2], [1, 3]]), [1, 3]) == 1

    def test_argfirst2D_not_found(self):
        assert utils.argfirst2D(np.array([[1, 2], [1, 3]]), [0, 0]) is None

    def test_shortest_paths_no_path(self):
        grid = np.array([
            # 0  1  2
            [1, 0, 1],  # 0
            [1, 0, 1],  # 1
            [0, 0, 1],  # 2
        ], dtype=bool)
        lookfor = ((0, 0), (2, 2)),
        paths = [[]]
        assert np.array_equal(paths, utils.get_shortest_paths(grid, lookfor))

    def test_json_numpy_encoder_int(self):
        assert (json.dumps(np.uint(10), cls=utils.JSONNumpyEncoder)
                == json.dumps(10))

    def test_json_numpy_encoder_bool(self):
        assert (json.dumps(np.bool_(True), cls=utils.JSONNumpyEncoder)
                == json.dumps(True))

    def test_json_numpy_encoder_float(self):
        assert (json.dumps(np.float32(10.0), cls=utils.JSONNumpyEncoder)
                == json.dumps(10.0))

    def test_json_numpy_encoder_int_array(self):
        array = np.arange(10, dtype=np.uint).reshape(2, 5)
        assert (json.dumps(array, cls=utils.JSONNumpyEncoder)
                == json.dumps(array.tolist()))

    def test_json_numpy_encoder_bool_array(self):
        array = np.ones((2, 5), dtype=np.bool_)
        assert (json.dumps(array, cls=utils.JSONNumpyEncoder)
                == json.dumps(array.tolist()))

    def test_json_numpy_encoder_float_array(self):
        array = np.arange(10, dtype=np.float).reshape(2, 5)
        assert (json.dumps(array, cls=utils.JSONNumpyEncoder)
                == json.dumps(array.tolist()))

    def test_json_numpy_encoder_tuple_array(self):
        assert (json.dumps((1, 2), cls=utils.JSONNumpyEncoder)
                == json.dumps((1, 2)))

    def test_json_numpy_encoder_dict_array(self):
        array = np.arange(10, dtype=np.float).reshape(2, 5)
        assert (json.dumps({'a': array}, cls=utils.JSONNumpyEncoder)
                == json.dumps({'a': array.tolist()}))

    def test_serialize_json(self):
        array = np.arange(10, dtype=np.uint).reshape(2, 5)
        assert (utils.serialize_json(array)
                == json.dumps(array.tolist()))

    def test_edges_to_graph(self):
        graph = nx.read_graphml(fixtures_path('graph.graphml'))
        with open(fixtures_path('graph.json'), 'r') as json_graph:
            edges = json.load(json_graph)
            out = nx.parse_graphml(utils.edges_to_graph(edges))
        assert nodeset(out) == nodeset(graph)
        assert edgeset(out) == edgeset(graph)

    def test_edges_to_graph_defaults_to_graphml(self):
        with open(fixtures_path('graph.json'), 'r') as json_graph:
            edges = json.load(json_graph)
            out = nx.parse_graphml(utils.edges_to_graph(edges))
            graph = nx.parse_graphml(utils.edges_to_graph(edges,
                                                          fmt='graphml'))
        assert nodeset(out) == nodeset(graph)
        assert edgeset(out) == edgeset(graph)

    def test_edges_to_graph_graphml(self):
        graph = nx.read_graphml(fixtures_path('graph.graphml'))
        with open(fixtures_path('graph.json'), 'r') as json_graph:
            edges = json.load(json_graph)
            out = nx.parse_graphml(utils.edges_to_graph(edges, fmt='graphml'))
        assert nodeset(out) == nodeset(graph)
        assert edgeset(out) == edgeset(graph)

    def test_edges_to_graph_gml(self):
        graph = nx.read_gml(fixtures_path('graph.gml'))
        with open(fixtures_path('graph.json'), 'r') as json_graph:
            edges = json.load(json_graph)
            out = nx.parse_gml(utils.edges_to_graph(edges, fmt='gml'))
        assert nodeset(out) == nodeset(graph)
        assert edgeset(out) == edgeset(graph)

    def test_edges_to_graph_gexf(self):
        graph = nx.read_gexf(fixtures_path('graph.gexf'))
        with open(fixtures_path('graph.json'), 'r') as json_graph:
            edges = json.load(json_graph)
            out = nx.read_gexf(io.StringIO(utils.edges_to_graph(edges,
                                                                fmt='gexf')))
        assert nodeset(out) == nodeset(graph)
        assert edgeset(out) == edgeset(graph)

    def test_edges_to_graph_edgelist(self):
        graph = nx.read_edgelist(fixtures_path('graph.edgelist'))
        with open(fixtures_path('graph.json'), 'r') as json_graph:
            edges = json.load(json_graph)
            out = nx.parse_edgelist(
                    utils.edges_to_graph(edges, fmt='edgelist').split('\n'))
        assert nodeset(out) == nodeset(graph)
        assert edgeset(out) == edgeset(graph)

    def test_edges_to_graph_nodelink(self):
        with open(fixtures_path('graph.nodelink.json')) as nodelink_graph:
            graph = nx_json_graph.node_link_graph(json.load(nodelink_graph))
        with open(fixtures_path('graph.json'), 'r') as json_graph:
            edges = json.load(json_graph)
            out = nx_json_graph.node_link_graph(
                utils.edges_to_graph(edges, fmt='nodelink'))
        assert nodeset(out) == nodeset(graph)
        assert edgeset(out) == edgeset(graph)
