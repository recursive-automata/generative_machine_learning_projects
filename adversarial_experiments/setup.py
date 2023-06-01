# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup

setup(
    name='gan_utils',
    version='0.1.0',
    description='Utilities for using adversarial losses',
    author='Simon Schneider',
    author_email='simon.r.schneider@gmail.com',
    license='MIT',
    package_dir={'': 'gan_utils'},
    packages=find_packages(where='gan_utils'),

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)