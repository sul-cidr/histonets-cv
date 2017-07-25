# -*- coding: utf-8 -*-
import base64
import collections
import gzip
import imghdr
import json
import locale
import os
import sys
import warnings
from itertools import chain
from urllib.parse import urlparse
from urllib.request import urlopen

import click
import cv2
import noteshrink
import numpy as np
import PIL
from click.utils import get_os_args
from skimage import filters as skfilters
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import minmax_scale
from sklearn.utils.validation import DataConversionWarning


# Constants
RAW = 'raw'
IMAGE = 'image'


class Stream(click.ParamType):
    """Click option type for http/https/file inputs

    Based on https://github.com/moshe/click-stream
    """
    name = 'stream'
    SUPPORTED_SCHEMES = ('http', 'https', 'file')

    def __init__(self, mode='r'):
        self.mode = mode

    def convert(self, param=None, ctx=None, value=None):
        if not value:
            content = sys.stdin.readlines()
        else:
            is_compressed = value.endswith('.gz')
            is_binary = self.mode.endswith('b')
            if os.path.exists(value):
                local_mode = 'rb' if is_compressed else self.mode
                file_obj = open(value, local_mode)
            else:
                url = urlparse(value)
                if url.scheme not in self.SUPPORTED_SCHEMES:
                    raise click.BadParameter(
                        'Scheme \'{}\' not supported'.format(url.scheme))
                else:
                    file_obj = urlopen(value)
            if is_compressed:
                with gzip.GzipFile(mode=self.mode, fileobj=file_obj) as file:
                    content = file.read()
                    if not is_binary:
                        content = local_decode(content)
            else:
                content = file_obj.read()
            file_obj.close()
        return content


class JSONStream(Stream):
    """JSON Stream Click option type to handle and decode JSON input and files
    coming (compressed or not) from the Internet (http:// and https://) or
    locally (file://, absolute, or relative paths).
    """

    def convert(self, param=None, ctx=None, value=None):
        try:
            return json.loads(value)
        except ValueError:
            content = super().convert(param=param, ctx=ctx, value=value)
            return json.loads(content)


class Choice(click.Choice):
    """Fix to click.Choice to be able to use integer choices"""

    def get_metavar(self, param):
        return '[%s]' % '|'.join(str(c) for c in self.choices)


class Image(object):
    """Proxy class to handle image input in the commands"""
    __slots__ = ('image', 'format')

    def __init__(self, content=None, image=None):
        self.image = None
        self.format = None
        if isinstance(image, Image):
            self.image = image.image
            self.format = image.format
        elif image is not None:
            self.image = image
        elif content:
            image_format = imghdr.what(file='', h=content)
            if image_format is not None:
                image_array = np.fromstring(content, np.uint8)
                self.image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                self.format = image_format
        if self.image is None:
            raise click.BadParameter('Image format not supported')

    @classmethod
    def get_images(cls, values):
        """Helper to process local, remote, and base64 piped images as input,
        and return Image objects"""
        images = []
        if not all(values):
            for value in sys.stdin.readlines():
                content = base64.b64decode(local_encode(value))
                images.append(cls(content))
        else:
            for value in values:
                if isinstance(value, cls):
                    image = value
                elif isinstance(value, np.ndarray):
                    image = cls(image=value)
                else:
                    try:
                        content = Stream(mode='rb').convert(value=value)
                    except Exception as e:
                        raise click.BadParameter(e)
                    image = cls(content)
                images.append(image)
        return images


def image_as_array(f):
    """Decorator to handle image as Image and as numpy array"""

    def wrapper(*args, **kwargs):
        image = kwargs.get('image', None) or (args and args[0])
        if isinstance(image, Image):
            if 'image' in kwargs:
                kwargs['image'] = image.image
            else:
                args = list(args)
                args[0] = image.image
                args = tuple(args)
        return f(*args, **kwargs)

    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper


def output_as_mask(f):
    """Decorator to add a return_mask option when image and mask are being
    returned"""

    def wrapper(*args, **kwargs):
        return_mask = kwargs.pop('return_mask', False)
        image, mask = f(*args, **kwargs)
        if return_mask:
            return mask
        else:
            return cv2.bitwise_and(image, image, mask=mask)

    wrapper.__name__ = f.__name__
    wrapper.__doc__ = """{}
    The binary mask can also be returned instead by setting
    return_mask to True.""".format(f.__doc__)
    return wrapper


def get_images(ctx, param, value):
    """Callback to retrieve images by either their local path or URL"""
    try:
        return Image.get_images(value)
    except Exception as e:
        raise click.BadParameter(e)


def parse_jsons(ctx, param, value):
    """Callback to load a list JSON strings as objects"""
    try:
        return [json.loads(v) for v in value]
    except ValueError:
        raise click.BadParameter("Polygon JSON malformed.")


def io_handler(input=None, *args, **kwargs):
    """Decorator to handle the 'input' argument and the 'output' option.
    If input is other than 'image', it is considered to be a JSON file or
    URL. Defaults to 'image'."""
    is_callable = callable(input)

    def decorator(f, *args, **kwargs):
        """Auxiliary decorator to allow io_handler be used with or without
        parameters"""

        def _get_image(ctx, param, value):
            """Helper to process only the first input image"""
            try:
                return Image.get_images([value])[0]  # return only first Image
            except Exception as e:
                raise click.BadParameter(e)

        if not is_callable and input != 'image':
            input_dec = click.argument(input, required=False,
                                       callback=JSONStream())
        else:
            input_dec = click.argument('image', required=False,
                                       callback=_get_image)

        @input_dec
        @click.option('--output', '-o', type=click.File('wb'),
                      help='File name to save the output. For images, if '
                           'the file extension is different than IMAGE, '
                           'a conversion is made. When not given, standard '
                           'output is used and images are serialized using '
                           'Base64; and to JSON otherwise.')
        def wrapper(*args, **kwargs):
            input_param = kwargs.get(input)
            output_param = kwargs.pop('output', None)
            ctx = click.get_current_context()
            result = ctx.invoke(f, *args, **kwargs)
            if output_param == RAW:
                return result
            result_is_image = isinstance(result, np.ndarray)
            if result_is_image:
                if output_param and '.' in output_param.name:
                    image_format = output_param.name.split('.')[-1]
                elif getattr(input_param, 'format', None):
                    image_format = input_param.format
                else:
                    image_format = 'png'  # default
                try:
                    _, result = cv2.imencode(".{}".format(image_format),
                                             result)
                except cv2.error:
                    raise click.BadParameter(
                        'Image format output not supported')
            else:
                result = local_encode(json.dumps(result))
            if output_param is not None:
                output_param.write(result)
                output_param.close()
            else:
                if result_is_image:
                    result = base64.b64encode(result)
                click.echo(result)

        # needed for click to work
        wrapper.__name__ = f.__name__
        new_line = '' if '\b' in f.__doc__ else '\b\n\n'
        if not is_callable and input != 'image':
            wrapper.__doc__ = (
                """{}{}\n    - {} path to a local (file://) or """
                """remote (http://, https://) JSON file representing {}.\n"""
                """      A JSON string can also be piped as input""".format(
                    f.__doc__, new_line, input.upper(), input))
        else:
            wrapper.__doc__ = (
                """{}{}\n    - IMAGE path to a local (file://) or """
                """remote (http://, https://) image file.\n"""
                """      A Base64 string can also be piped as input """
                """image.""".format(
                    f.__doc__, new_line))
        return wrapper

    if is_callable:
        return decorator(input)
    else:
        return decorator


def pair_options_to_argument(argument, options, args=None, args_slice=None):
    """Enforces pairing of options to an argument. Only commands with one
    argument with nargs=-1 are supported. Not paired options do still work.

    Options is a dictionary with the option name as key and the default value
    as value. A slice to specify where in the arguments the argument and the
    options are found can be used. By default it will ignore first and last.

    Example::

        @click.command()
        @click.argument('arg', nargs=-1, required=True)
        @click.option('-o', '--option', multiple=True)
        @pair_options_to_argument('arg', {'option': 0})
        def command(arg, option):
            pass
    """
    pairings = list(options.values())
    args_slice = args_slice or (1, -1)
    _args = args

    def func(f):
        def wrapper(*args, **kwargs):
            ctx = click.get_current_context()
            append = ''
            os_args = []
            for os_arg in (_args or get_os_args())[slice(*args_slice)]:
                if os_arg.startswith('-'):
                    append = os_arg
                else:
                    os_args.append(("{}\b".format(append), os_arg))
                    append = ''
            params = {}
            defaults = {}
            for param in ctx.command.get_params(ctx):
                if param.name in options.keys():
                    params[param.name] = param.opts
                    defaults[param.name] = param.default
            _kwargs = {k: v for k, v in kwargs.items() if k in pairings}
            _params = {k: {} for k, v in params.items()}
            if pairings and os_args:
                index = 0
                for prefix, _ in os_args:
                    if prefix.startswith('\b'):
                        index += 1
                    else:
                        for name, opts in params.items():
                            if any(prefix.startswith(opt) for opt in opts):
                                kw_arg_index = len(_params[name])
                                kw_arg = kwargs[name][kw_arg_index]
                                _params[name][index - 1] = kw_arg
                for name, opts in _params.items():
                    default = defaults[name] or options[name]
                    _list = [default] * len(kwargs[argument])
                    for k, v in opts.items():
                        _list[k] = v
                    _kwargs[name] = tuple(_list)
                kwargs.update(_kwargs)
            if all(len(kwargs[opt]) == 0 for opt in options):
                for name, opts in _params.items():
                    default = defaults[name] or options[name]
                    _list = [default] * len(kwargs[argument])
                    for k, v in opts.items():
                        _list[k] = v
                    _kwargs[name] = tuple(_list)
                kwargs.update(_kwargs)
            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        return wrapper
    return func


def local_encode(value):
    """Encode a string to bytes by using the system preferred encoding.
    Defaults to utf8 otherwise."""
    encoding = locale.getpreferredencoding(False)
    try:
        return value.encode(encoding)
    except LookupError:
        return value.encode('utf8', errors='replace')


def local_decode(value):
    """Decode bytes into a string by using the system preferred encoding.
    Defaults to utf8 otherwise."""
    encoding = locale.getpreferredencoding(False)
    try:
        return value.decode(encoding)
    except LookupError:
        return value.decode('utf8', errors='replace')


def parse_pipeline_json(ctx, param, value):
    """Parse the actions JSON used mainly in the pipeline command"""
    try:
        obj = json.loads(value)
    except Exception:
        raise click.BadParameter('Malformed JSON')
    if (not isinstance(obj, (tuple, list))
            or not all(map(lambda x: isinstance(x, dict), obj))):
        raise click.BadParameter('Malformed JSON')
    for action in obj:
        if 'action' not in action:
            raise click.BadParameter('Missing key for action')
    return obj


@image_as_array
def get_color_histogram(image):
    """Calculate the color histogram of image (colors and their counts)"""
    colors = np.reshape(image, (np.prod(image.shape[:2]), 3)).tolist()
    return collections.Counter([tuple(color) for color in colors])


@image_as_array
def get_palette(pixel_colors, n_colors, background_value=25,
                background_saturation=20, method=None):
    """Calculate a palette of n_colors from RGB values from an array of colors.
    Parameters background_value and background_saturation are ignored for
    methods other than 'auto'.
    When method='auto', the first palette entry is always the background
    color; the rest are determined from foreground pixels by running K-Means
    clustering.
    Returns the palette."""
    if method is None:
        method = 'auto'
    if method == 'auto':
        options = collections.namedtuple(
            'options', ['quiet', 'value_threshold', 'sat_threshold']
        )(
            quiet=True,
            value_threshold=background_value / 100.0,
            sat_threshold=background_saturation / 100.0,
        )
        # 8 bits per channel avoid quantization artifacts,
        # such as converting [255, 255, 255] to [254, 254, 254],
        bg_color = noteshrink.get_bg_color(pixel_colors, bits_per_channel=8)
        fg_mask = noteshrink.get_fg_mask(bg_color, pixel_colors, options)
        if fg_mask.any():
            masked_image = pixel_colors[fg_mask]
        else:
            masked_image = pixel_colors
        centers, _ = kmeans(masked_image, n_colors - 1)
        # We need to guarantee that the first color returned is bg_color
        # and that colors are all unique
        bg_color = np.array(bg_color, dtype=np.uint8)
        palette = np.unique(centers.astype(np.uint8), axis=0).astype(np.uint8)
        palette = palette[np.all(palette - bg_color, axis=1)]
        return np.vstack((bg_color, palette))
    else:
        if method == 'kmeans':
            palette, _ = kmeans(pixel_colors, n_colors)
        else:
            img = PIL.Image.fromarray(np.array([pixel_colors])[:, :, ::-1],
                                      mode='RGB')
            quant = img.quantize(colors=n_colors,
                                 method=get_quantize_method(method))
            quant_palette = quant.getpalette()[:3 * n_colors]
            palette = np.array(quant_palette).reshape((n_colors, 3))
        return unique(palette, axis=0).astype(np.uint8)


def get_quantize_method(method):
    """Transform a string ('median', 'octree', 'linear', 'max') to the
    corresponding PIL quantize method constant"""
    if method == 'median':
        return PIL.Image.MEDIANCUT
    elif method == 'octree':
        return PIL.Image.FASTOCTREE
    else:  # method in ('linear', 'max', 'maxcoverage'):
        return PIL.Image.MAXCOVERAGE


def unique(array, axis=None):
    """Like np.unique but preserving order of first apparition"""
    uniques, indices = np.unique(array, axis=axis, return_index=True)
    return uniques[np.argsort(indices)]


def kmeans(X, n_clusters, **kwargs):
    """Classify vectors in X using K-Means algorithm with n_clusters.
    Arguments in kwargs are passed to scikit-learn MiniBatchKMeans.
    Returns a tuple of cluster centers and predicted labels."""
    clf = MiniBatchKMeans(n_clusters=n_clusters, **kwargs)
    labels = clf.fit_predict(X)
    centers = clf.cluster_centers_.astype(np.ubyte)
    return centers, labels


def get_mask_polygons(polygons, height, width):
    """Turn a list of polygons into a mask image of height by width.
    Each polygon is expressed as a list of [x, y] points."""
    mask = np.zeros((height, width), dtype=np.ubyte)
    cv2.fillPoly(mask, np.int32(polygons), color=255)
    return mask


@image_as_array
def match_template_mask(image, template, mask=None, method=None, sigma=0.33):
    """Match template against image applying mask to template using method.
    Method can be either of (None, 'laplacian', 'sobel', 'scharr', 'prewitt',
    'roberts', 'canny').
    Returns locations to look for max values."""
    if mask is not None:
        if method:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel)
            if method == 'laplacian':
                # use CV_64F to not loose edges, convert to uint8 afterwards
                edge_image = np.uint8(np.absolute(
                    cv2.Laplacian(image, cv2.CV_64F)))
                edge_template = np.uint8(np.absolute(
                    cv2.Laplacian(template, cv2.CV_64F)
                ))
            elif method in ('sobel', 'scharr', 'prewitt', 'roberts'):
                filter_func = getattr(skfilters, method)
                edge_image = filter_func(image)
                edge_template = filter_func(template)
                edge_image = convert(edge_image)
                edge_template = convert(edge_template)
            else:  # method == 'canny'
                values = np.hstack([image.ravel(), template.ravel()])
                median = np.median(values)
                lower = int(max(0, (1.0 - sigma) * median))
                upper = int(min(255, (1.0 + sigma) * median))
                edge_image = cv2.Canny(image, lower, upper)
                edge_template = cv2.Canny(template, lower, upper)
            results = cv2.matchTemplate(edge_image, edge_template & mask,
                                        cv2.TM_CCOEFF_NORMED)
        else:
            results = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED,
                                        mask)
    else:
        results = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return results


def parse_colors(ctx, param, value):
    """Callback to parse color values from a JSON list or hexadecimal string
    to a RGB tuple.
    """
    colors = []
    for color in value:
        if isinstance(color, (list, tuple)):
            r, g, b = color
        elif color.startswith('#'):
            hex_color = color[1:]
            hex_color_len = len(hex_color)
            r, g, b = None, None, None
            try:
                if hex_color_len == 3:
                    r, g, b = [int(''.join([l] * 2), 16) for l in hex_color]
                elif hex_color_len == 6:
                    r, g, b = list(map(lambda x: int(''.join(x), 16),
                                       zip(*[iter(hex_color)] * 2)))
            except ValueError:
                raise click.BadParameter(
                    "Malformed hex color: {}".format(color))
        else:
            try:
                r, g, b = json.loads(color)
            except ValueError:
                raise click.BadParameter(
                    "Malformed JSON or hexadecimal string: {}".format(color))
        if all(isinstance(c, int) and 0 <= c <= 255 for c in (r, g, b)):
            colors.append((r, g, b))
        else:
            raise click.BadParameter(
                "Invalid color value: {}".format(color))
    return colors


def parse_palette(ctx, param, value):
    """Callback to turn a JSON representing a palette of colors in hexadecimal
    or by its RGB components, into a list of all RGB components"""
    if value:
        try:
            colors = parse_colors(ctx, param, json.loads(value))
            return np.array(colors, np.uint8)
        except ValueError:
            raise click.BadParameter(
                "Malformed JSON palette: {}".format(value))


def convert(image):
    """Convert a scikit-image binary image matrix to OpenCV"""
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings('ignore', category=DataConversionWarning)
        return minmax_scale(image, (0, 255)).astype(np.ubyte)


def parse_histogram(histogram):
    """Parse a dictionary or JSON string representing a histogram of colors
    by parsing the keys that codify colors into lists of RGB components
    and the values to integer numbers"""
    if not isinstance(histogram, dict):
        histogram = json.loads(histogram)
    try:
        return {parse_colors(None, None, [k])[0]: int(v)
                for k, v in histogram.items()}
    except ValueError as e:
        raise click.BadParameter("Malformed histogram: {}".format(e))


def sample_histogram(histogram, sample_fraction=0.05):
    """Sample a sample_fraction of colors from histogram"""
    colors = np.fromiter(histogram.keys(), dtype='i8,i8,i8', count=-1)
    counts = np.fromiter(histogram.values(), dtype=np.float32, count=-1)
    size = counts.sum()
    weights = counts / size
    if sample_fraction is None:
        samples = (k * v for k, v in histogram.items())
        fraction = size.astype(int)
    else:
        fraction = int(size * sample_fraction)
        samples = np.random.choice(colors, size=fraction, p=weights)
    unraveled = np.fromiter(chain.from_iterable(samples), np.uint8, count=-1)
    return unraveled.reshape(fraction, 3)
