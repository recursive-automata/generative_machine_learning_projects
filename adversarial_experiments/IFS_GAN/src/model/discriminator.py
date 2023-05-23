# -*- coding: utf-8 -*-

import tensorflow as tf

_leaky_relu = tf.keras.layers.LeakyReLU


def build_discriminator(n_dim, width, depth=5, dropout_frac=0.0):
    inputs = tf.keras.layers.Input((n_dim,))

    x = inputs
    for _ in range(depth):
        x = tf.keras.layers.Dense(units=width)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = _leaky_relu(0.2)(x)
    x = tf.keras.layers.Dropout(dropout_frac)(x)
    x = tf.keras.layers.Dense(units=1)(x)

    return tf.keras.models.Model(inputs, x)


def calculate_gradient_penalty(discriminator, random_generator, x, x_, training=True):
    alpha = random_generator.uniform([tf.shape(x)[0], 1])
    interpolated = (1 - alpha) * x + alpha * x_

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        y = discriminator(interpolated, training=training)

    grads = gp_tape.gradient(y, [interpolated])[0]
    norm_sq = tf.reduce_sum(tf.square(grads), axis=[1])
    return 0.5 * norm_sq
