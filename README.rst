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

binarize
~~~~~~~~
Usage: histonets binarize [OPTIONS] [IMAGE]

Binarize IMAGE using a thresholding method.

Example::

  histonets binarize -m otsu file://...


- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -m, --method [sauvola|isodata|otsu|li]
                                  Thresholding method to obtain the binary
                                  image. For reference, see http://scikit-imag
                                  e.org/docs/dev/auto_examples/xx_applications
                                  /plot_thresholding.html. Defaults to 'li'.
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

blobs
~~~~~
Usage: histonets blobs [OPTIONS] [IMAGE]

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
Usage: histonets brightness [OPTIONS] VALUE [IMAGE]

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
Usage: histonets clean [OPTIONS] [IMAGE]

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
  -p, --palette TEXT              Local file, URL, or JSON string representing
                                  a palette of colors encoded as lists of RGB
                                  components or hexadecimal strings preceded
                                  by the hash character (#). Ex: '["#fa4345",
                                  "[123, 9, 108]", [1, 2, 3]]'. If a palette
                                  is passed in, colors are ignored.
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

contrast
~~~~~~~~
Usage: histonets contrast [OPTIONS] VALUE [IMAGE]

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
Usage: histonets denoise [OPTIONS] VALUE [IMAGE]

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
Usage: histonets download [OPTIONS] [IMAGE]

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
Usage: histonets enhance [OPTIONS] [IMAGE]

Clean IMAGE automatically with sane defaults.

- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -p, --palette TEXT     Local file, URL, or JSON string representing a
                         palette of colors encoded as lists of RGB components
                         or hexadecimal strings preceded by the hash character
                         (#). Ex: '["#fa4345", "[123, 9, 108]", [1, 2, 3]]'.
  -o, --output FILENAME  File name to save the output. For images, if the file
                         extension is different than IMAGE, a conversion is
                         made. When not given, standard output is used and
                         images are serialized using Base64; and to JSON
                         otherwise.
  

equalize
~~~~~~~~
Usage: histonets equalize [OPTIONS] VALUE [IMAGE]

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
  

graph
~~~~~
Usage: histonets graph [OPTIONS] REGIONS [IMAGE]

Build a undirected graph using the center points of REGIONS as nodes
and the paths in the binary grid expressed in IMAGE as edges.

Example::

  histonets graph '[[[50,50],[120,50]],[[120, 82],[50,82]]]' -sm vw file://

- REGIONS is a path to a local (file://) or remote (http://, https://) JSON
          file representing a list of bounding boxes expressed as two
          [x, y] coordinates points in pixels with regards to IMAGE,
          one for the top-left corner and a second for the bottom-left one.
          For example, '[[[50, 50], [120, 50]], [[120, 82], [50, 82]]]' is
          a list that contains two regions.

- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -sm, --simplification-method [rdp|vw]
                                  Specify the line simplification algorithm to
                                  reduce the number of pixels in each path.
                                  Available algorithms are 'rdp' for the
                                  Ramer–Douglas–Peucker's algorithm, and 'vw'
                                  for Visvalingam–Whyatt's algorithm. Defaults
                                  to 'vw'.
  -st, --simplification-tolerance INTEGER RANGE
                                  Exponent of the inverse simplification
                                  method tolerance, e.g., 3 involves a
                                  tolerance of 10^(-3)). Ranges from 0 to 10.
                                  Defaults to 0.
  -f, --format [edgelist|gexf|gml|graphml|nodelink]
                                  Format to save the graph in. All formats are
                                  derived from NetworkX's "Reading and Writing
                                  graphs": http://networkx.readthedocs.io/en/s
                                  table/reference/readwrite.html. Defaults to
                                  'graphml'
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

match
~~~~~
Usage: histonets match [OPTIONS] TEMPLATES... [IMAGE]

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
  

palette
~~~~~~~
Usage: histonets palette [OPTIONS] [HISTOGRAM]

Extract a palette of colors from HISTOGRAM.

- HISTOGRAM path to local file, URL, or JSON string representing a
  dictionary with colors as keys and the count (pixels) of those colors as
  values. Colors can be given as a list of its RGB components, or
  in hexadecimal format preceded by the hash character (#).

  Example::

    histonets palette '{"#fa4345": 3829, "[123, 9, 108]": 982}'

- HISTOGRAM path to a local (file://) or remote (http://, https://) JSON file representing histogram.
  A JSON string can also be piped as input

Options:

  -c, --colors INTEGER RANGE      Number of output colors. Ranges from 2 to
                                  128. Defaults to 8.
  -m, --method [auto|kmeans|median|linear|max|octree]
                                  Method for computing the palette. 'auto'
                                  runs an optimized K-Means algorithm by
                                  sampling the histogram and detecting the
                                  background color first; 'kmeans' performs a
                                  clusterization of the existing colors using
                                  the K-Means algorithm; 'median' refers to
                                  the median cut algorithm;  'max' runs a
                                  maximum coverage process (also aliased as
                                  'linear'); and 'octree' executes a fast
                                  octree quantization algorithm. Defaults to
                                  'auto'.
  -f, --sample-fraction INTEGER RANGE
                                  Percentage of pixels to sample. Ranges from
                                  0 to 100. Defaults to 5.
  -bv, --background-value INTEGER RANGE
                                  Threshold value to consider a pixel
                                  background. Ranges from 0 to 100. Defaults
                                  to 25.
  -bs, --background-saturation INTEGER RANGE
                                  Threshold saturation to consider a pixel
                                  background. Ranges from 0 to 100. Defaults
                                  to 20.
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

pipeline
~~~~~~~~
Usage: histonets pipeline [OPTIONS] ACTIONS [IMAGE]

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
Usage: histonets posterize [OPTIONS] [COLORS] [IMAGE]

Posterize IMAGE by reducing its number of colors.

- COLORS, the number of colors of the output image, ranges from 0 to 64.
- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -m, --method [kmeans|median|linear|max|octree]
                                  Method for computing the palette. 'kmeans'
                                  performs a clusterization of the existing
                                  colors using the K-Means algorithm; 'median'
                                  refers to the median cut algorithm;  'max'
                                  runs a maximum coverage process (also
                                  aliased as 'linear'); and 'octree' executes
                                  a fast octree quantization algorithm.
                                  Defaults to 'kmeans'.
  -p, --palette TEXT              Local file, URL, or JSON string representing
                                  a palette of colors encoded as lists of RGB
                                  components or hexadecimal strings preceded
                                  by the hash character (#). Ex: '["#fa4345",
                                  "[123, 9, 108]", [1, 2, 3]]'. If a palette
                                  is passed in, colors are ignored.
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

ridges
~~~~~~
Usage: histonets ridges [OPTIONS] [IMAGE]

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
  -d, --dilation INTEGER RANGE    Dilation radius to thicken the mask of
                                  detected ridges. Ranges from 0 to 100.
                                  Defaults to 1.
  -m, --mask                      Returns a black and white mask instead.
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

select
~~~~~~
Usage: histonets select [OPTIONS] COLORS... [IMAGE]

Select COLORS in IMAGE, turning the rest into black.

Example::

  histonets select "[225, 47, 90]" "#8ad70e" -t 80  file://...

- COLOR is a JSON string representing a color as a list of
        its RGB components or a hexadecimal string starting
        with #.
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
  

skeletonize
~~~~~~~~~~~
Usage: histonets skeletonize [OPTIONS] [IMAGE]

Extract the morphological skeleton of IMAGE. If the image is not black
and white, it will be binarized using a binarization-method, which by
default it's Li's algorithm (see the binarize command).
The black and white image can also be thickened (dilated) by adjusting
the dilation parameter before extracting the skeleton image.

Example::

  histonets skeletonize -m thin -d 0 -b otsu file://...


- IMAGE path to a local (file://) or remote (http://, https://) image file.
  A Base64 string can also be piped as input image.

Options:

  -m, --method [3d|combined|medial|regular|thin]
                                  Method to extract the topological skeleton
                                  of IMAGE. For reference, see http://scikit-i
                                  mage.org/docs/dev/auto_examples/xx_applicati
                                  ons/plot_thresholding.html. Defaults to a
                                  'combined' approach of '3d', 'medial', and
                                  'regular'.
  -d, --dilation INTEGER RANGE    Dilation radius to thicken the binarized
                                  image prior to perform skeletonization.
                                  Ranges from 0 to 100. Defaults to 6.
  -b, --binarization-method [sauvola|isodata|otsu|li]
                                  Thresholding method to obtain the binary
                                  image. For reference, see http://scikit-imag
                                  e.org/docs/dev/auto_examples/xx_applications
                                  /plot_thresholding.html. Defaults to 'li'.
  -i, --invert                    Invert the black and white colors of the
                                  binary image prior to skeletonization.
  -o, --output FILENAME           File name to save the output. For images, if
                                  the file extension is different than IMAGE,
                                  a conversion is made. When not given,
                                  standard output is used and images are
                                  serialized using Base64; and to JSON
                                  otherwise.
  

smooth
~~~~~~
Usage: histonets smooth [OPTIONS] VALUE [IMAGE]

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

