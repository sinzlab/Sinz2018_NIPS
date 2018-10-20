#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='nips2018',
    version='0.0.0',
    description='Spatial Transformer Networks for Neuroscience Data',
    author='Fabian Sinz',
    author_email='sinz@bcm.edu',
    url='https://github.com/sinzlab/Sinz2018_NIPS',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm', 'gitpython', 'python-twitter', 'scikit-image', 'datajoint', 'atflow', 'attorch',
                      'h5py'],
)
