# -*- coding: utf-8 -*-

import tensorflow as tf


def calculate_hausdorff_suprema(x, y):
    """Given two collections of samples `x` and `y`, calculate the supremum distances between the two collections.
    To do this, we first calculate the minimum distance from each element to a point in the opposite collection,
    then take the maximum of these minimum distances.

    The first supremum distance provides a measure of how close `y` is to being included  in `x` as a subset,
    whereas the second supremum distance provides a measure of how close `x` is to being included  in `y` as a subset.
    The [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance) is the greater of the two suprema.

    :param x: a tensor of type float and shape [_, n_dim]
    :param y: a tensor of type float and shape [_, n_dim]
    :return: the two supremum distances
    """
    all_distances_sq = y[:, None] - x[None, :]
    all_distances_sq = tf.square(all_distances_sq)
    all_distances_sq = tf.reduce_sum(all_distances_sq, axis=2)
    all_distances = tf.sqrt(all_distances_sq + 1e-8)

    min_distances_x = tf.reduce_min(all_distances, axis=1)
    min_distances_y = tf.reduce_min(all_distances, axis=0)

    max_min_distance_x = tf.reduce_max(min_distances_x, axis=0)
    max_min_distance_y = tf.reduce_max(min_distances_y, axis=0)

    return max_min_distance_x, max_min_distance_y
