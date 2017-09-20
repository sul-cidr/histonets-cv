#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pip
import pkg_resources
from setuptools import Extension, setup


def ensure(requirement):
    try:
        pkg_resources.require(requirement)
    except pkg_resources.ResolutionError:
        pip.main(['install', requirement])


ensure('numpy>=1.13.0')
ensure('Cython==0.26')
import numpy
from Cython.Build import cythonize


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.lock') as requirements_file:
    requirements = requirements_file.readlines()

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='histonets',
    version='0.1.0',
    description="Computer vision part of the Histonets project",
    long_description=readme + '\n\n' + history,
    author="Javier de la Rosa",
    author_email='versae@gmail.com',
    url='https://github.com/sul-cidr/histonets-cv',
    packages=[
        'histonets',
    ],
    package_dir={'histonets':
                 'histonets'},
    entry_points={
        'console_scripts': [
            'histonets=histonets.cli:main'
        ]
    },
    ext_modules=cythonize(
        [Extension('*', ['histonets/*.pyx'])],
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False,
            'embedsignature': True,
            'infer_types': True,
        }
    ),
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='histonets',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
