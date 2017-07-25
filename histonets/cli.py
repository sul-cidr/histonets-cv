# -*- coding: utf-8 -*-
import sys

import click
import cv2

from .utils import (
    Choice,
    Image,
    get_images,
    get_mask_polygons,
    io_handler,
    pair_options_to_argument,
    parse_colors,
    parse_histogram,
    parse_jsons,
    parse_palette,
    parse_pipeline_json,
    RAW,
)
from .api import (
    adjust_brightness,
    adjust_contrast,
    binarize_image,
    denoise_image,
    histogram_equalization,
    histogram_palette,
    smooth_image,
    color_reduction,
    auto_clean,
    match_templates,
    select_colors,
    remove_ridges,
    remove_blobs,
    skeletonize_image,
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
            click.echo(command.get_usage(ctx)
                              .replace('histonets ', "histonets {} ".format(
                                       command_name))
                              .replace('\b\n', ''))
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
@click.argument("actions", callback=parse_pipeline_json)
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
        command = main.get_command(ctx, action['action'])
        if command is None:
            raise click.BadParameter(
                "Action '{}' not found".format(action['action']))
        action_options = action.get('options', {})
        options = {param.name: action_options.get(param.name, param.default)
                   for param in command.params[:-2]}
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
@click.option('-c', '--colors', type=click.IntRange(2, 128),
              default=8,
              help='Number of output colors. Ranges from 2 to 128. '
                   'Defaults to 8.')
@click.option('-m', '--method',
              type=click.Choice(['auto', 'kmeans', 'median', 'linear', 'max',
                                 'octree']),
              default='auto',
              help='Method for computing the palette. \'auto\' runs an '
                   'optimized K-Means algorithm by sampling the histogram and '
                   'detecting the background color first; \'kmeans\' performs '
                   'a clusterization of the existing colors using the K-Means '
                   'algorithm; \'median\' refers to the median cut algorithm; '
                   ' \'max\' runs a maximum coverage process (also aliased '
                   'as \'linear\'); and \'octree\' executes a fast octree '
                   'quantization algorithm. Defaults to \'auto\'.')
@click.option('-f', '--sample-fraction', type=click.IntRange(1, 100),
              default=5,
              help='Percentage of pixels to sample. Ranges from 0 to 100. '
                   'Defaults to 5.')
@click.option('-bv', '--background-value', type=click.IntRange(1, 100),
              default=25,
              help='Threshold value to consider a pixel background. '
                   'Ranges from 0 to 100. Defaults to 25.')
@click.option('-bs', '--background-saturation', type=click.IntRange(1, 100),
              default=20,
              help='Threshold saturation to consider a pixel background. '
                   'Ranges from 0 to 100. Defaults to 20.')
@io_handler('histogram')
def palette(histogram, colors, method, sample_fraction, background_value,
            background_saturation):
    """Extract a palette of colors from HISTOGRAM.

    \b
    - HISTOGRAM path to local file, URL, or JSON string representing a
      dictionary with colors as keys and the count (pixels) of those colors as
      values. Colors can be given as a list of its RGB components, or
      in hexadecimal format preceded by the hash character (#).

      Example::

        histonets palette '{"#fa4345": 3829, "[123, 9, 108]": 982}'
    """
    histogram = parse_histogram(histogram)
    return histogram_palette(
        histogram, n_colors=colors, method=method,
        sample_fraction=sample_fraction, background_value=background_value,
        background_saturation=background_saturation).tolist()


@main.command()
@click.argument("colors", required=False, type=click.IntRange(2, 128))
@click.option('-m', '--method',
              type=click.Choice(['kmeans', 'median', 'linear', 'max',
                                 'octree']),
              default='kmeans',
              help='Method for computing the palette. \'kmeans\' performs '
                   'a clusterization of the existing colors using the K-Means '
                   'algorithm; \'median\' refers to the median cut algorithm; '
                   ' \'max\' runs a maximum coverage process (also aliased '
                   'as \'linear\'); and \'octree\' executes a fast octree '
                   'quantization algorithm. Defaults to \'kmeans\'.')
@click.option('-p', '--palette', callback=parse_palette,
              default=None,
              help='Local file, URL, or JSON string representing a palette of '
                   'colors encoded as lists of RGB components or '
                   'hexadecimal strings preceded by the hash character (#). '
                   'Ex: \'["#fa4345", "[123, 9, 108]", [1, 2, 3]]\'. '
                   'If a palette is passed in, colors are ignored.')
@io_handler
def posterize(image, colors, method, palette):
    """Posterize IMAGE by reducing its number of colors.

    \b
    - COLORS, the number of colors of the output image, ranges from 0 to 64."""
    return color_reduction(image, colors, method, palette)


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
@click.option('-p', '--palette', callback=parse_palette,
              default=None,
              help='Local file, URL, or JSON string representing a palette of '
                   'colors encoded as lists of RGB components or '
                   'hexadecimal strings preceded by the hash character (#). '
                   'Ex: \'["#fa4345", "[123, 9, 108]", [1, 2, 3]]\'. '
                   'If a palette is passed in, colors are ignored.')
@io_handler
def clean(image, background_value, background_saturation, colors,
          sample_fraction, white_background, saturate, palette):
    """Clean IMAGE automatically with sane defaults and allows for parameter
    fine tunning."""
    return auto_clean(image, background_value, background_saturation,
                      colors, sample_fraction, white_background, saturate,
                      palette)


@main.command()
@click.option('-p', '--palette', callback=parse_palette,
              default=None,
              help='Local file, URL, or JSON string representing a palette of '
                   'colors encoded as lists of RGB components or '
                   'hexadecimal strings preceded by the hash character (#). '
                   'Ex: \'["#fa4345", "[123, 9, 108]", [1, 2, 3]]\'.')
@io_handler
def enhance(image, palette):
    """Clean IMAGE automatically with sane defaults."""
    return auto_clean(image, palette=palette)


@main.command()
@click.argument('templates', nargs=-1, required=True, callback=get_images)
@click.option('-th', '--threshold', type=click.IntRange(0, 100),
              multiple=True,
              help='Threshold to match TEMPLATE to IMAGE. '
                   'Ranges from 0 to 100. Defaults to 80.')
@click.option('-f', '--flip', type=click.Choice(
                ['horizontal', 'h', 'vertical', 'v', 'both', 'b', 'all', 'a']),
              multiple=True,
              help='Whether also match TEMPLATE flipped horizontally. '
                   'vertically, or both. Defaults to not flipping.')
@click.option('-e', '--exclude-regions', callback=parse_jsons,
              multiple=True,
              help='JSON list of polygons expressed as [x, y] points to '
                   'specify regions to cut out when matching. '
                   'For example, [[[50,50],[120,50],[120,82],[50,82]]] '
                   'is a list that contains one single polygon.')
@io_handler
@pair_options_to_argument('templates', {
    'threshold': 80,
    'flip': None,
    'exclude_regions': None,
})
def match(image, templates, threshold, flip, exclude_regions):
    """Look for TEMPLATES in IMAGE and return the bounding boxes of
    the matches. Options may be provided after each TEMPLATE.

    Example::

      histonets match http://foo.bar/tmpl1 -th 50 http://foo.bar/tmpl2 -th 95

    \b
    - TEMPLATE is a path to a local (file://) or remote (http://, https://)
      image file of the template to look for."""
    # TODO: Click invoke fails at testing time, but not at runtime :(
    #       template options should be a list of the same length that templates
    none_list = [None] * len(templates)
    args = (
        Image.get_images(templates),  # pipeline does not invoke the decorator
        threshold or none_list,
        flip or none_list,
        exclude_regions or none_list,
    )
    if len(set(len(x) for x in args)) != 1:
        raise click.BadParameter('Some templates or options are missing.')
    image_templates = []
    for (template_image, template_threshold, template_flip,
         template_exclude_regions) in zip(*args):
        mask = None
        if template_exclude_regions:
            try:
                mask = ~get_mask_polygons(template_exclude_regions,
                                          *template_image.image.shape[:2])
            except cv2.error:
                raise click.BadParameter('Polygons JSON is malformed.')
        image_templates.append({
            'image': template_image.image,
            'threshold': template_threshold,
            'flip': template_flip,
            'mask': mask,
        })
    matches = match_templates(image, image_templates)
    return matches.tolist()


@main.command()
@click.argument('colors', nargs=-1, required=True, callback=parse_colors)
@click.option('-t', '--tolerance', type=click.IntRange(0, 100),
              multiple=True,
              help='Tolerance to match COLOR in IMAGE. '
                   'Ranges from 0 to 100. Defaults to 0 (exact COLOR).')
@click.option('-m', '--mask', is_flag=True,
              help='Returns a black and white mask instead.')
@io_handler
@pair_options_to_argument('colors', {
    'tolerance': 0,
})
def select(image, colors, tolerance, mask):
    """Select COLORS in IMAGE, turning the rest into black.

    Example::

      histonets select "[225, 47, 90]" "#8ad70e" -t 80  file://...

    \b
    - COLOR is a JSON string representing a color as a list of
            its RGB components or a hexadecimal string starting
            with #."""
    return select_colors(image, zip(*(colors, tolerance)), return_mask=mask)


@main.command()
@click.option('-w', '--width', type=click.IntRange(1, 100),
              default=6,
              help='Width in pixels of the ridges to detect. '
                   'Ranges from 1 to 100. Defaults to 6.')
@click.option('-th', '--threshold', type=click.IntRange(0, 255),
              default=128,
              help='Threshold to binarize detected ridges. '
                   'Ranges from 0 to 255. Defaults to 128.')
@click.option('-d', '--dilation', type=click.IntRange(0, 100),
              default=3,
              help='Dilation to thicken the mask of detected ridges. '
                   'Ranges from 0 to 100. Defaults to 3.')
@click.option('-m', '--mask', is_flag=True,
              help='Returns a black and white mask instead.')
@io_handler
def ridges(image, width, threshold, dilation, mask):
    """Remove ridges from IMAGE, turning them into black.

    Example::

      histonets ridges --width 6 file://...
    """
    return remove_ridges(image, width, threshold, dilation, return_mask=mask)


@main.command()
@click.option('-min', '--minimum-area', type=int,
              default=0,
              help='Minimum area in pixels of the white blobs to detect. '
                   'Defaults to 0.')
@click.option('-max', '--maximum-area', type=int,
              default=sys.maxsize,
              help="Maximum area in pixels of the white blobs to detect. "
                   "Defaults to {}.".format(sys.maxsize))
@click.option('-th', '--threshold', type=click.IntRange(0, 255),
              default=128,
              help='Threshold to binarize before detecting blobs. '
                   'Ranges from 0 to 255. Defaults to 128.')
@click.option('-c', '--connectivity', type=Choice([4, 8, 16]),
              help='Connectivity method to consider blobs boundaries. '
                   'It can take adjacent pixels in a 4 pixels cross '
                   'neighborhood (top, right, bottom, left), 8 pixels (all '
                   'around), or 16 pixels (anti-aliased). '
                   'Defaults to 4 neighbors.')
@click.option('-m', '--mask', is_flag=True,
              help='Returns a black and white mask instead.')
@io_handler
def blobs(image, minimum_area, maximum_area, threshold, connectivity, mask):
    """Binarize using threshold and remove white blobs of contiguous pixels
    of size between min and max from IMAGE, turning them into black.

    Example::

      histonets blobs -max 100 -c 8 file://...
    """
    method = "{}-connected".format(connectivity)
    return remove_blobs(image, minimum_area, maximum_area, threshold, method,
                        return_mask=mask)


@main.command()
@click.option('-m', '--method',
              type=Choice(['sauvola', 'isodata', 'otsu', 'li']),
              default='li',
              help='Thresholding method to obtain the binary image. '
                   'For reference, see http://scikit-image.org/docs/dev'
                   '/auto_examples/xx_applications/plot_thresholding.html. '
                   'Defaults to \'li\'.')
@io_handler
def binarize(image, method):
    """Binarize IMAGE using a thresholding method.

    Example::

      histonets binarize -m otsu file://...
    """
    return binarize_image(image, method)


@main.command()
@click.option('-m', '--method',
              type=Choice(['3d', 'combined', 'medial', 'regular', 'thin']),
              default='combined',
              help='Method to extract the topological skeleton of IMAGE. '
                   'For reference, see http://scikit-image.org/docs/dev'
                   '/auto_examples/xx_applications/plot_thresholding.html. '
                   'Defaults to a \'combined\' approach of \'3d\', '
                   '\'medial\', and \'regular\'.')
@click.option('-d', '--dilation', type=click.IntRange(0, 100),
              default=13,
              help='Dilation to thicken the binarized image prior to perform '
                   'skeletonization. '
                   'Ranges from 0 to 100. Defaults to 13.')
@click.option('-b', '--binarization-method',
              type=Choice(['sauvola', 'isodata', 'otsu', 'li']),
              default='li',
              help='Thresholding method to obtain the binary image. '
                   'For reference, see http://scikit-image.org/docs/dev'
                   '/auto_examples/xx_applications/plot_thresholding.html. '
                   'Defaults to \'li\'.')
@io_handler
def skeletonize(image, method, dilation, binarization_method):
    """Extract the morphological skeleton of IMAGE. If the image is not black
    and white, it will be binarized using a binarization-method, which by
    default it's Li's algorithm (see the binarize command).
    The black and white image can also be thickened (dilated) by adjusting
    the dilation parameter before extracting the skeleton image.

    Example::

      histonets skeletonize -m thin -d 0 -b otsu file://...
    """
    return skeletonize_image(image, method, dilation, binarization_method)


if __name__ == "__main__":
    main()  # pragma: no cover
