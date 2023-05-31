# -*- coding: utf-8 -*-

import tensorflow as tf

_leaky_relu = tf.keras.layers.LeakyReLU


class ReparameterizeLayer(tf.keras.layers.Layer):
    def __init__(self, random_generator):
        super(ReparameterizeLayer, self).__init__()
        self.random_generator = random_generator

    @tf.function
    def call(self, args, training=True):
        """Executes the reparameterization trick, commonly used in VAEs

        :param args: a tuple of two tensors (means, log_vars)
        :param training: bool, whether to run in training mode (ie, non-deterministically)
        :return: in training mode, a sample from the multivariate normal distribution;
        in non-training mode, the mean of the random variable
        """
        if training:
            noise = self.random_generator.normal(tf.shape(args[1]))
            return args[0] + tf.math.exp(0.5 * args[1]) * noise
        else:
            return args[0]


def build_convnet_encoder(
        random_generator,
        pixels,
        channels,
        latent_dim,
        width=512,
        depth=4,
        kernel_size=3,
        dropout_frac=0.0
):
    inputs = tf.keras.layers.Input((pixels, pixels, channels))

    x = inputs
    width = int(width / 2 ** depth)
    for _ in range(depth):
        width *= 2
        x = tf.keras.layers.Conv2D(width, kernel_size=kernel_size, strides=2, padding='same')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout_frac)(x)

    mean = tf.keras.layers.Dense(latent_dim)(x)
    log_var = tf.keras.layers.Dense(latent_dim)(x)
    x = ReparameterizeLayer(random_generator)([mean, log_var])

    return tf.keras.models.Model(inputs, x)


def build_convnet_decoder(
        pixels,
        channels,
        latent_dim,
        width=512,
        depth=4,
        kernel_size=3,
        dropout_frac=0.0
):
    inputs = tf.keras.layers.Input((latent_dim,))

    initial_pixels = int(pixels / 2 ** depth)
    x = tf.keras.layers.Dense(width * initial_pixels ** 2)(inputs)
    for _ in range(depth):
        width = int(width / 2)
        x = tf.keras.layers.Conv2DTranspose(width, kernel_size=kernel_size, strides=2, padding='same')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_frac)(x)
    logits = tf.keras.layers.Conv2D(channels, kernel_size=1)(x)
    pixels = tf.keras.layers.Activation('sigmoid')(logits)

    return tf.keras.models.Model(inputs, [logits, pixels])


class Autoencoder(tf.keras.Model):
    """A keras model for training Autoencoders.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x, training=True):
        z_ = self.encoder(x, training=training)
        x_ = self.decoder(z_, training=training)
        return z_, x_

    def update(self, loss, tape, optimizer):
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
