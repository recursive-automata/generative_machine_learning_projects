# -*- coding: utf-8 -*-

import numpy as np


def make_2D_rotation(theta):
    matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    def _2D_rotation(x):
        return x @ matrix.T
    return _2D_rotation


def make_2D_translation(delta_x, delta_y):
    delta = np.array([[delta_x, delta_y]])
    def _2D_translation(x):
        return x + delta
    return _2D_translation


def make_2D_rescale(factor):
    def _2D_rescale(x):
        return x * factor
    return _2D_rescale


def compose(transformations):
    def _apply(x):
        y = x
        for t in transformations:
            y = t(y)
        return y
    return _apply
