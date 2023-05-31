# -*- coding: utf-8 -*-

import tensorflow as tf

_leaky_relu = tf.keras.layers.LeakyReLU


def build_mlp(n_dim, width, depth=5, dropout_frac=0.0, n_outputs=1):
    inputs = tf.keras.layers.Input((n_dim,))

    x = inputs
    for _ in range(depth):
        x = tf.keras.layers.Dense(units=width)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = _leaky_relu(0.2)(x)
    x = tf.keras.layers.Dropout(dropout_frac)(x)
    x = tf.keras.layers.Dense(units=n_outputs)(x)

    return tf.keras.models.Model(inputs, x)


def build_convnet(pixels, channels, width=512, depth=4, kernel_size=3, dropout_frac=0.0, n_outputs=1):
    inputs = tf.keras.layers.Input((pixels, pixels, channels))

    x = inputs
    width = int(width / 2 ** depth)
    for _ in range(depth):
        width *= 2
        x = tf.keras.layers.Conv2D(width, kernel_size=kernel_size, strides=2, padding='same')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = _leaky_relu(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout_frac)(x)
    x = tf.keras.layers.Dense(units=n_outputs)(x)

    return tf.keras.models.Model(inputs, x)


def calculate_gradient_penalty(discriminator, random_generator, data, generated, training=True, mode='interpolate'):
    assert mode in ['data', 'generated', 'interpolate']

    if mode == 'data':
        inputs = data
    elif mode == 'generate':
        inputs = generated
    else:
        alpha = random_generator.uniform([tf.shape(data)[0], 1])
        inputs = (1 - alpha) * data + alpha * generated

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(inputs)
        y = discriminator(inputs, training=training)

    grads = gp_tape.gradient(y, [inputs])[0]
    grads = tf.reshape(grads, [tf.shape(grads)[0], -1])
    norm_sq = tf.reduce_sum(tf.square(grads), axis=[1])
    return 0.5 * norm_sq
