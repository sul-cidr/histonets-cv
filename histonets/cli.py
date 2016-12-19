# -*- coding: utf-8 -*-
import click

from .utils import io_handler, pair_options_to_argument, parse_json, RAW
from .histonets import (
    adjust_brightness,
    adjust_contrast,
    denoise_image,
    histogram_equalization,
    smooth_image,
    color_reduction,
    auto_clean,
)


@click.group(invoke_without_command=True)
@click.option('--rst', is_flag=True, help='Show help in ReST format.')
@click.version_option()
def main(rst=None):
    """Histonets computer vision application for image processing"""
    ctx = click.get_current_context()
    if rst:
        click.echo()
        comamnds_text = 'Commands'
        options_text = 'Options:'
        main_help, _ = main.get_help(ctx).split(comamnds_text, 1)
        click.echo(main_help)
        click.echo(comamnds_text)
        click.echo('-' * len(comamnds_text))
        click.echo()
        for command_name, command in sorted(main.commands.items()):
            click.echo(command_name)
            click.echo('~' * len(command_name))
            click.echo(command.get_usage(ctx).replace('\b\n', ''))
            click.echo()
            click.echo(command.help.replace('\b\n', ''))
            command_help = command.get_help(ctx)
            _, command_options_help = command_help.split(options_text, 1)
            command_options, _ = command_options_help.rsplit('--help', 1)
            click.echo()
            click.echo(options_text)
            click.echo(command_options)
            click.echo()
    elif ctx.invoked_subcommand is None:
        click.echo(main.get_help(ctx))


@main.command()
@click.argument("actions", callback=parse_json)
@io_handler
def pipeline(image, actions):
    """Allow chaining a series of actions to be applied to IMAGE.
    Output will depend on the last action applied.

    \b
    - ACTIONS is a JSON list of dictionaries containing each an 'action' key
      specifying the action to apply, a 'arguments' key which is a
      list of arguments, and a 'options' key with a dictionary to set the
      options for the corresponding action.

      Example::

        histonets pipeline '[{"action": "contrast", "options": {"value": 50}}]'
    """
    output = image.image
    for action in actions:
        ctx = click.get_current_context()
        arguments = [output] + action.get('arguments', [])
        options = action.get('options', {})
        command = main.get_command(ctx, action['action'])
        if command is None:
            raise click.BadParameter(
                "Action '{}' not found".format(action['action']))
        options['output'] = RAW
        try:
            output = command.callback(*arguments, **options)
        except TypeError as e:
            raise click.BadParameter(e)
    return output


@main.command()
@io_handler
def download(image):
    """Download IMAGE."""
    return image.image


@main.command()
@click.argument("value", type=click.IntRange(0, 200))
@io_handler
def contrast(image, value):
    """Adjust contrast of IMAGE.

    \b
    - VALUE ranges from 0 to 200."""
    return adjust_contrast(image, value)


@main.command()
@click.argument("value", type=click.IntRange(0, 200))
@io_handler
def brightness(image, value):
    """Adjust brightness of IMAGE.

    \b
    - VALUE ranges from 0 to 200."""
    return adjust_brightness(image, value)


@main.command()
@click.argument("value", type=click.IntRange(0, 100))
@io_handler
def smooth(image, value):
    """Smooth IMAGE using bilateral filter.

    \b
    - VALUE ranges from 0 to 100."""
    return smooth_image(image, value)


@main.command()
@click.argument("value", type=click.IntRange(0, 100))
@io_handler
def equalize(image, value):
    """Histogram equalization on IMAGE.

    \b
    - VALUE ranges from 0 to 100."""
    return histogram_equalization(image, value)


@main.command()
@click.argument("value", type=click.IntRange(0, 100))
@io_handler
def denoise(image, value):
    """Denoise IMAGE.

    \b
    - VALUE ranges from 0 to 100."""
    return denoise_image(image, value)


@main.command()
@click.argument("colors", type=click.IntRange(2, 128))
@click.option('-m', '--method', type=click.Choice(['kmeans', 'linear']),
              default='kmeans',
              help='Method for computing the palette. \'kmeans\' performs '
                   'a clusterization of the existing colors using the K-Means '
                   'algorithm; \'linear\' tries to quantize colors in a '
                   'linear scale, therefore will approximate to the next '
                   'power of 2. Defaults to \'kmeans\'.')
@io_handler
def posterize(image, colors, method):
    """Posterize IMAGE by reducing its number of colors.

    \b
    - COLORS, the number of colors of the output image, ranges from 0 to 64."""
    return color_reduction(image, colors, method)


@main.command()
@click.option('-bv', '--background-value', type=click.IntRange(1, 100),
              default=25,
              help='Threshold value to consider a pixel background. '
                   'Ranges from 0 to 100. Defaults to 25.')
@click.option('-bs', '--background-saturation', type=click.IntRange(1, 100),
              default=20,
              help='Threshold saturation to consider a pixel background. '
                   'Ranges from 0 to 100. Defaults to 20.')
@click.option('-c', '--colors', type=click.IntRange(2, 128),
              default=8,
              help='Number of output colors. Ranges from 2 to 128. '
                   'Defaults to 8.')
@click.option('-f', '--sample-fraction', type=click.IntRange(1, 100),
              default=5,
              help='Percentage of pixels to sample. Ranges from 0 to 100. '
                   'Defaults to 5.')
@click.option('-w', '--white-background', is_flag=True,
              default=False,
              help='Make background white.')
@click.option('-s/-ns', '--saturate/--no-saturate',
              default=True,
              help='Saturate colors (default).')
@io_handler
def clean(image, background_value, background_saturation, colors,
          sample_fraction, white_background, saturate):
    """Clean IMAGE automatically with sane defaults and allows for parameter
    fine tunning."""
    return auto_clean(image, background_value, background_saturation,
                      colors, sample_fraction, white_background, saturate)


@main.command()
@io_handler
def enhance(image):
    """Clean IMAGE automatically with sane defaults."""
    return auto_clean(image)


@main.command()
@click.argument('templates', nargs=-1, required=True)
@click.option('-th', '--threshold', type=click.IntRange(0, 100),
              multiple=True,
              help='Threshold to match TEMPLATE to IMAGE. '
                   'Ranges from 0 to 100. Defaults to 80.')
@io_handler
@pair_options_to_argument('templates', {'threshold': 80})
def match(image, templates, threshold):
    """Look for TEMPLATES in IMAGE and return the bounding boxes of
    the matches. Options may be provided after each TEMPLATE.

    Example::

      histonets match http://foo.bar/tmpl1 -th 50 http://foo.bar/tmpl2 -th 95

    \b
    - TEMPLATE is a path to a local (file://) or remote (http://, https://)
      image file of the template to look for."""
    if len(set(len(x) for x in (templates, threshold))) != 1:
        raise click.BadParameter('Some options are missing.')
    _templates = []
    for template, template_threshold in zip(*[templates, threshold]):
        _templates.append({
            'template': template,
            'options': {
                'threshold': template_threshold,
            }
        })
    click.echo(_templates)


if __name__ == "__main__":
    main()
