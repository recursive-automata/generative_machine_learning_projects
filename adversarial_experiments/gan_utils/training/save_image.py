# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def scatter(x, filename, xlim=None, ylim=None, figsize=(12, 6), axis='off', **kwargs):
    """Save a scatter plot to file.

    :param x: A numpy array of type float and shape `(_, 2)`
    :param filename:
    :param xlim:
    :param ylim:
    :param axis:
    :param figsize:
    """
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if axis is not None:
        plt.axis(axis)

    plt.scatter(*x.T, **kwargs)
    plt.savefig(filename)

    plt.close()
    del fig


def raster(f, xlim, ylim, filename=None, pixels=250, figsize=(12, 6)):
    """Save an image of a 2D raster to file.

    :param f: a function from (float, float) -> float
    :param xlim:
    :param ylim:
    :param filename:
    :param pixels:
    :param figsize:
    """
    min_x = xlim[0]
    delta_x = xlim[1] - xlim[0]
    min_y = ylim[0]
    delta_y = ylim[1] - ylim[0]

    x_grid = np.array([
        [min_x + delta_x * i / (pixels - 1), min_y + delta_y * j / (pixels - 1)]
        for i in range(pixels)
        for j in range(pixels)
    ])
    c = f(x_grid)

    fig = plt.figure(figsize=figsize)
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(x_grid[:, 0], x_grid[:, 1], c=c)
    plt.savefig(filename)

    plt.close()
    del fig
    del x_grid
    del c
