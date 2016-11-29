# -*- coding: utf-8 -*-
import base64
import errno
import imghdr
import io
import json
import locale
import os
import stat
import sys

import click
import cv2
import numpy as np
import requests
from requests.adapters import BaseAdapter
from requests.compat import urlparse, unquote


class FileAdapter(BaseAdapter):
    def send(self, request, **kwargs):
        """ Adapted from https://github.com/dashea/requests-file:
        Wraps a file, described in request, in a Response object.

            :param request: The PreparedRequest` being "sent".
            :returns: a Response object containing the file
        """
        if request.method not in ('GET', 'HEAD'):
            error_msg = "Invalid request method {}".format(request.method)
            raise ValueError(error_msg)
        url_parts = urlparse(request.url)
        resp = requests.Response()
        try:
            path_parts = []
            if url_parts.netloc:
                # Local files are interpreted as netloc
                path_parts = [unquote(p) for p in url_parts.netloc.split('/')]
            path_parts += [unquote(p) for p in url_parts.path.split('/')]
            while path_parts and not path_parts[0]:
                path_parts.pop(0)
            if any(os.sep in p for p in path_parts):
                raise IOError(errno.ENOENT, os.strerror(errno.ENOENT))
            if path_parts and (path_parts[0].endswith('|') or
                               path_parts[0].endswith(':')):
                path_drive = path_parts.pop(0)
                if path_drive.endswith('|'):
                    path_drive = path_drive[:-1] + ':'
                while path_parts and not path_parts[0]:
                    path_parts.pop(0)
            else:
                path_drive = ''
            path = os.path.join(path_drive, *path_parts)
            if path_drive and not os.path.splitdrive(path):
                path = os.path.join(path_drive, *path_parts)
            resp.raw = io.open(path, 'rb')
            resp.raw.release_conn = resp.raw.close
        except IOError as e:
            if e.errno == errno.EACCES:
                resp.status_code = requests.codes.forbidden
            elif e.errno == errno.ENOENT:
                resp.status_code = requests.codes.not_found
            else:
                resp.status_code = requests.codes.bad_request
            resp_str = str(e).encode(locale.getpreferredencoding(False))
            resp.raw = io.BytesIO(resp_str)
            resp.headers['Content-Length'] = len(resp_str)
            resp.raw.release_conn = resp.raw.close
        else:
            resp.status_code = requests.codes.ok
            resp_stat = os.fstat(resp.raw.fileno())
            if stat.S_ISREG(resp_stat.st_mode):
                resp.headers['Content-Length'] = resp_stat.st_size
        return resp

    def close(self):
        pass


class Image(object):
    """Proxy class to handle image input in the commands"""
    __slots__ = ('image', 'format')

    def __init__(self, content):
        self.image = None
        self.format = None
        if content:
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
            session = requests.Session()
            session.mount('file://', FileAdapter())
            for value in values:
                try:
                    response = session.get(value)
                    if response.status_code == requests.codes.ok:
                        content = response.content
                    else:
                        content = None
                except Exception as e:
                    raise click.BadParameter(e)
                images.append(cls(content))
            session.close()
        return images


def local_encode(value):
    encoding = locale.getpreferredencoding(False)
    try:
        return value.encode(encoding)
    except LookupError:
        return value.encode('utf8', errors='replace')


def io_handler(f, *args, **kwargs):
    """Decorator to handle the 'image' argument and the 'output' option"""

    def _get_image(ctx, param, value):
        """Helper to process only the first input image"""
        try:
            return Image.get_images([value])[0]  # return only first Image
        except Exception as e:
            raise click.BadParameter(e)

    @click.argument('image', required=False, callback=_get_image)
    @click.option('--output', '-o', type=click.File('wb'),
                  help='File name to save the output. For images, if '
                       'the file extension is different than IMAGE, '
                       'a conversion is made. When not given, standard '
                       'output is used and images are serialized using '
                       'Base64; and to JSON otherwise.')
    def wrapper(*args, **kwargs):
        image = kwargs.get('image') or args[0]
        output = kwargs.pop('output')
        ctx = click.get_current_context()
        result = ctx.invoke(f, *args, **kwargs)
        result_is_image = isinstance(result, np.ndarray)
        if result_is_image:
            if output and '.' in output.name:
                image_format = output.name.split('.')[-1]
            else:
                image_format = image.format
            try:
                _, result = cv2.imencode(".{}".format(image_format), result)
            except cv2.error:
                raise click.BadParameter('Image format output not supported')
        else:
            result = local_encode(json.dumps(result))
        if output is not None:
            output.write(result)
            output.close()
        else:
            if result_is_image:
                result = base64.b64encode(result)
            click.echo(result)

    # needed for click to work
    wrapper.__name__ = f.__name__
    new_line = '' if '\b' in f.__doc__ else '\b\n\n'
    wrapper.__doc__ = (
        """{}{}\n    - IMAGE path to a local (file://) or """
        """remote (http://, https://) image file.\n"""
        """      A Base64 string can also be piped as input image.""".format(
            f.__doc__, new_line))
    return wrapper
