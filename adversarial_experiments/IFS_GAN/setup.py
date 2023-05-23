# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup

setup(
    name='ifs_gan',
    version='0.1.0',
    description='Fitting an iterated function system from examples using an adversarial loss',
    author='Simon Schneider',
    author_email='simon.r.schneider@gmail.com',
    license='MIT',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)