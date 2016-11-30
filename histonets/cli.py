# -*- coding: utf-8 -*-
import click

from .utils import io_handler, parse_json, RAW
from .histonets import adjust_contrast, adjust_brightness, smooth_image


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
        output = command.callback(*arguments, **options)
    return output


@main.command()
@io_handler
def download(image):
    """Download IMAGE."""
    return image.image


@main.command()
@click.argument("value", type=click.IntRange(-100, 100))
@io_handler
def contrast(image, value):
    """Adjust contrast of IMAGE.

    \b
    - VALUE ranges from -100 to 100."""
    return adjust_contrast(image, value)


@main.command()
@click.argument("value", type=click.IntRange(-100, 100))
@io_handler
def brightness(image, value):
    """Adjust brightness of IMAGE.

    \b
    - VALUE ranges from -100 to 100."""
    return adjust_brightness(image, value)


@main.command()
@click.argument("value", type=click.IntRange(0, 100))
@io_handler
def smooth(image, value):
    """Smooth IMAGE using bilateral filter.

    \b
    - VALUE ranges from 0 to 100."""
    return smooth_image(image.image, value)


if __name__ == "__main__":
    main()
