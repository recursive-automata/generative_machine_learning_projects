# -*- coding: utf-8 -*-

import numpy as np

from .transforms2D import *

def koch_iterate(x):
    """Applies the Iterated Function System associated with the Koch curve
    to a given collection of points in the plane

    :param x: a float array of shape (n_obs, 2)
    :return: a float array of shape (4 * n_obs, 2)
    """
    functions = [
        make_2D_rescale(1/3),
        compose([
            make_2D_rescale(1/3),
            make_2D_rotation(np.pi / 3),
            make_2D_translation(1/3, 0)
        ]),
        compose([
            make_2D_rescale(1/3),
            make_2D_rotation(-np.pi / 3),
            make_2D_translation(1/2, np.sqrt(3)/6)
        ]),
        compose([
            make_2D_rescale(1/3),
            make_2D_translation(2/3, 0)
        ])
    ]
    return np.vstack([f(x) for f in functions])


def make_koch_samples(n_iters=6, starting_points=None):
    """Generate a collection of points that approximate the Koch curve.

    :param n_iters: How many iterations of the Iterated Function System to apply
    :param starting_points: Optional, a float array of shape (n_obs, 2)
    :return: a float array of shape (n_obs * 4^n_iters, 2)
    """
    starting_points = np.array([[1.0, 0], [0, 0]]) if starting_points is None else starting_points

    koch = starting_points
    for _ in range(n_iters):
        koch = koch_iterate(koch)
    koch = koch - koch.mean(axis=0, keepdims=True)
    maximum = np.abs(koch).max()
    koch = koch / maximum
    return koch
