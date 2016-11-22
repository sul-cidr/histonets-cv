# -*- coding: utf-8 -*-
import click

from .utils import open_image
from .histonets import adjust_contrast


@click.group()
def main():
    """Histonets computer vision application for image processing"""


@main.command()
@click.argument("image", callback=open_image)
def download(image):
    click.echo(image.image)


@main.command()
@click.argument("image", callback=open_image)
@click.argument("value", type=click.IntRange(-100, 100))
def contrast(image, value):
    adjust_contrast(image, value)


if __name__ == "__main__":
    main()
