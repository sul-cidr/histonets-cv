# -*- coding: utf-8 -*-
import click

from .utils import open_image


@click.group()
def main():
    """Histonets computer vision application for image processing"""


@main.command()
@click.argument("image", callback=open_image)
def download(image):
    click.echo(image.image)


if __name__ == "__main__":
    main()
