# -*- coding: utf-8 -*-
import click

from .utils import io_handler
from .histonets import adjust_contrast


@click.group()
def main():
    """Histonets computer vision application for image processing"""


@main.command()
@io_handler
def download(image):
    return image.image


@main.command()
@io_handler
@click.argument("value", type=click.IntRange(-100, 100))
def contrast(image, value):
    adjust_contrast(image, value)


@main.command()
@io_handler
@click.argument("value", type=click.IntRange(-100, 100))
def contrast(image, value):
    adjust_contrast(image, value)


if __name__ == "__main__":
    main()
