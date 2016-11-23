# -*- coding: utf-8 -*-
import base64
import errno
import imghdr
import io
import json
import locale
import stat
import os

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
    """Proxy class to hanle image input in the commands"""
    __slots__ = ('url', 'response', 'image', 'format')

    def __init__(self, url, response):
        self.url = url
        self.response = response
        self.image = None
        self.format = None
        if response.status_code == requests.codes.ok:
            image_format = imghdr.what(file='', h=response.content)
            if image_format is not None:
                image_array = np.fromstring(response.content, np.uint8)
                self.image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                self.format = image_format
        if self.image is None:
            raise click.BadParameter('Image format not supported')


def io_handler(f, *args, **kwargs):
    """Decorator to handle the 'image' argument and the 'output' option"""

    def get_image(ctx, param, value):
        """Helper that adds a file:// adapter and returns an Image object"""
        session = requests.Session()
        session.mount('file://', FileAdapter())

        if value is not None:
            try:
                response = session.get(value)
                return Image(value, response)
            except Exception as e:
                raise click.BadParameter(e)

    @click.argument('image', callback=get_image)
    @click.option('--output', '-o', type=click.File('wb'))
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
            encoding = locale.getpreferredencoding(False)
            try:
                result = json.dumps(result).encode(encoding)
            except LookupError:
                result = json.dumps(result).encode('utf8')
        if output is not None:
            output.write(result)
        else:
            if result_is_image:
                result = base64.b64encode(result)
            click.echo(result)
        output.close()

    wrapper.__name__ = f.__name__  # needed for click to work
    return wrapper
