===============================
Histonets Computer Vision
===============================


.. image:: https://img.shields.io/pypi/v/histonets.svg
        :target: https://pypi.python.org/pypi/histonets

.. image:: https://img.shields.io/travis/sul-cidr/histonets-cv.svg
        :target: https://travis-ci.org/sul-cidr/histonets-cv

.. image:: https://readthedocs.org/projects/histonets/badge/?version=latest
        :target: https://histonets.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/sul-cidr/histonets-cv/shield.svg
     :target: https://pyup.io/repos/github/sul-cidr/histonets-cv/
     :alt: Updates

.. image:: https://codecov.io/gh/sul-cidr/histonets-cv/branch/master/graph/badge.svg
     :target: https://codecov.io/gh/sul-cidr/histonets-cv
     :alt: Test Coverage

Computer vision part of the Histonets project


* Free software: Apache Software License 2.0
* Documentation: https://histonets.readthedocs.io.


Features
--------

.. commands_start

Usage: histonets [OPTIONS] COMMAND [ARGS]...

  Histonets computer vision application for image processing

Options:
  --rst      Show help in ReST format.
  --version  Show the version and exit.
  --help     Show this message and exit.


Commands
--------

blobs
~~~~~
Usage: histonets [OPTIONS] [IMAGE]

Binarize using threshold and remove white blobs of contiguous pixels
of size between min and max from IMAGE, turning them into black.

Example::

  histonets blobs -max 100 -c 8 file://...


- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -min, --minimum-area INTEGER    Minimum area in pixels of the white blobs to
                                  detect. Defaults to 0.
  -max, --maximum-area INTEGER    Maximum area in pixels of the white blobs to
                                  detect. Defaults to 9223372036854775807.
  -th, --threshold INTEGER RANGE  Threshold to binarize before detecting
                                  blobs. Ranges from 0 to 255. Defaults to
                                  128.
  -c, --connectivity [4|8|16]     Connectivity method to consider blobs
                                  boundaries. It can take adjacent pixels in a
                                  4 pixels cross neighborhood (top, right,
                                  bottom, left), 8 pixels (all around), or 16
                                  pixels (anti-aliased). Defaults to 4
                                  neighbors.
  -m, --mask                      Returns a black and white mask instead.
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

brightness
~~~~~~~~~~
Usage: histonets [OPTIONS] VALUE [IMAGE]

Adjust brightness of IMAGE.

- VALUE ranges from 0 to 200.
- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -o, --output FILENAME  File name to save the output. For images, if the file
                         extension is different than IMAGE, a conversion is
                         made. When not given, standard output is used and
                         images are serialized using Base64; and to JSON
                         otherwise.
  

clean
~~~~~
Usage: histonets [OPTIONS] [IMAGE]

Clean IMAGE automatically with sane defaults and allows for parameter
fine tunning.

- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -bv, --background-value INTEGER RANGE
                                  Threshold value to consider a pixel
                                  background. Ranges from 0 to 100. Defaults
                                  to 25.
  -bs, --background-saturation INTEGER RANGE
                                  Threshold saturation to consider a pixel
                                  background. Ranges from 0 to 100. Defaults
                                  to 20.
  -c, --colors INTEGER RANGE      Number of output colors. Ranges from 2 to
                                  128. Defaults to 8.
  -f, --sample-fraction INTEGER RANGE
                                  Percentage of pixels to sample. Ranges from
                                  0 to 100. Defaults to 5.
  -w, --white-background          Make background white.
  -s, --saturate / -ns, --no-saturate
                                  Saturate colors (default).
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

contrast
~~~~~~~~
Usage: histonets [OPTIONS] VALUE [IMAGE]

Adjust contrast of IMAGE.

- VALUE ranges from 0 to 200.
- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -o, --output FILENAME  File name to save the output. For images, if the file
                         extension is different than IMAGE, a conversion is
                         made. When not given, standard output is used and
                         images are serialized using Base64; and to JSON
                         otherwise.
  

denoise
~~~~~~~
Usage: histonets [OPTIONS] VALUE [IMAGE]

Denoise IMAGE.

- VALUE ranges from 0 to 100.
- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -o, --output FILENAME  File name to save the output. For images, if the file
                         extension is different than IMAGE, a conversion is
                         made. When not given, standard output is used and
                         images are serialized using Base64; and to JSON
                         otherwise.
  

download
~~~~~~~~
Usage: histonets [OPTIONS] [IMAGE]

Download IMAGE.

- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -o, --output FILENAME  File name to save the output. For images, if the file
                         extension is different than IMAGE, a conversion is
                         made. When not given, standard output is used and
                         images are serialized using Base64; and to JSON
                         otherwise.
  

enhance
~~~~~~~
Usage: histonets [OPTIONS] [IMAGE]

Clean IMAGE automatically with sane defaults.

- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -o, --output FILENAME  File name to save the output. For images, if the file
                         extension is different than IMAGE, a conversion is
                         made. When not given, standard output is used and
                         images are serialized using Base64; and to JSON
                         otherwise.
  

equalize
~~~~~~~~
Usage: histonets [OPTIONS] VALUE [IMAGE]

Histogram equalization on IMAGE.

- VALUE ranges from 0 to 100.
- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -o, --output FILENAME  File name to save the output. For images, if the file
                         extension is different than IMAGE, a conversion is
                         made. When not given, standard output is used and
                         images are serialized using Base64; and to JSON
                         otherwise.
  

match
~~~~~
Usage: histonets [OPTIONS] TEMPLATES... [IMAGE]

Look for TEMPLATES in IMAGE and return the bounding boxes of
the matches. Options may be provided after each TEMPLATE.

Example::

  histonets match http://foo.bar/tmpl1 -th 50 http://foo.bar/tmpl2 -th 95

- TEMPLATE is a path to a local (file://) or remote (http://, https://)
  image file of the template to look for.
- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -th, --threshold INTEGER RANGE  Threshold to match TEMPLATE to IMAGE. Ranges
                                  from 0 to 100. Defaults to 80.
  -f, --flip [horizontal|h|vertical|v|both|b|all|a]
                                  Whether also match TEMPLATE flipped
                                  horizontally. vertically, or both. Defaults
                                  to not flipping.
  -e, --exclude-regions TEXT      JSON list of polygons expressed as [x, y]
                                  points to specify regions to cut out when
                                  matching. For example,
                                  [[[50,50],[120,50],[120,82],[50,82]]] is a
                                  list that contains one single polygon.
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

pipeline
~~~~~~~~
Usage: histonets [OPTIONS] ACTIONS [IMAGE]

Allow chaining a series of actions to be applied to IMAGE.
Output will depend on the last action applied.

- ACTIONS is a JSON list of dictionaries containing each an 'action' key
  specifying the action to apply, a 'arguments' key which is a
  list of arguments, and a 'options' key with a dictionary to set the
  options for the corresponding action.

  Example::

    histonets pipeline '[{"action": "contrast", "options": {"value": 50}}]'

- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -o, --output FILENAME  File name to save the output. For images, if the file
                         extension is different than IMAGE, a conversion is
                         made. When not given, standard output is used and
                         images are serialized using Base64; and to JSON
                         otherwise.
  

posterize
~~~~~~~~~
Usage: histonets [OPTIONS] COLORS [IMAGE]

Posterize IMAGE by reducing its number of colors.

- COLORS, the number of colors of the output image, ranges from 0 to 64.
- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -m, --method [kmeans|linear]  Method for computing the palette. 'kmeans'
                                performs a clusterization of the existing
                                colors using the K-Means algorithm; 'linear'
                                tries to quantize colors in a linear scale,
                                therefore will approximate to the next power
                                of 2. Defaults to 'kmeans'.
  -o, --output FILENAME         File name to save the output. For images, if
                                the file extension is different than IMAGE, a
                                conversion is made. When not given, standard
                                output is used and images are serialized using
                                Base64; and to JSON otherwise.
  

ridges
~~~~~~
Usage: histonets [OPTIONS] [IMAGE]

Remove ridges from IMAGE, turning them into black.

Example::

  histonets ridges --width 6 file://...


- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -w, --width INTEGER RANGE       Width in pixels of the ridges to detect.
                                  Ranges from 1 to 100. Defaults to 6.
  -th, --threshold INTEGER RANGE  Threshold to binarize detected ridges.
                                  Ranges from 0 to 255. Defaults to 128.
  -d, --dilation INTEGER RANGE    Dilation to thicken the mask of detected
                                  ridges. Ranges from 0 to 100. Defaults to 3.
  -m, --mask                      Returns a black and white mask instead.
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

select
~~~~~~
Usage: histonets [OPTIONS] COLORS... [IMAGE]

Select COLORS in IMAGE, turning the rest into black.

Example::

  histonets select "[225, 47, 90]" "[124, 230, 7]" -t 80  file://...

- COLOR is a JSON string representing a color as a list of
        its RGB components.
- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -t, --tolerance INTEGER RANGE  Tolerance to match COLOR in IMAGE. Ranges
                                 from 0 to 100. Defaults to 0 (exact COLOR).
  -m, --mask                     Returns a black and white mask instead.
  -o, --output FILENAME          File name to save the output. For images, if
                                 the file extension is different than IMAGE, a
                                 conversion is made. When not given, standard
                                 output is used and images are serialized
                                 using Base64; and to JSON otherwise.
  

smooth
~~~~~~
Usage: histonets [OPTIONS] VALUE [IMAGE]

Smooth IMAGE using bilateral filter.

- VALUE ranges from 0 to 100.
- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -o, --output FILENAME  File name to save the output. For images, if the file
                         extension is different than IMAGE, a conversion is
                         made. When not given, standard output is used and
                         images are serialized using Base64; and to JSON
                         otherwise.
  


.. commands_end

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

