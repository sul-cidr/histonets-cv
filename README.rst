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
                                  background. . Ranges from 0 to 100. Defaults
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

