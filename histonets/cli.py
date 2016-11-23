# -*- coding: utf-8 -*-
import click

from .utils import io_handler


@click.group()
@click.version_option()
def main():
    """Histonets computer vision application for image processing"""


@main.command()
@io_handler
def download(image):
    return image.image


if __name__ == '__main__':
    main()
