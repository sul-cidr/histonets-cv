# -*- coding: utf-8 -*-
import click

from .utils import io_handler
from .histonets import adjust_contrast, adjust_brightness


@click.group()
@click.version_option()
def main():
    """Histonets computer vision application for image processing"""


@main.command()
@io_handler
def download(image):
    return image.image


@main.command()
@click.argument("value", type=click.IntRange(-100, 100))
@io_handler
def contrast(image, value):
    return adjust_contrast(image.image, value)


@main.command()
@click.argument("value", type=click.IntRange(-100, 100))
@io_handler
def brightness(image, value):
    return adjust_brightness(image.image, value)


if __name__ == "__main__":
    main()
