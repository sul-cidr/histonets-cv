#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_histonets
----------------------------------

Tests for `histonets` module.
"""
import os
import unittest
from click.testing import CliRunner

from histonets import cli


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
