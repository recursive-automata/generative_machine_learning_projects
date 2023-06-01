# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .common import update_model
from .discriminator import calculate_gradient_penalty
from .losses import continuous_bernoulli_loss


def build_convnet_encoder(
        pixels,
        channels,
        latent_dim,
        width=512,
        depth=4,
        kernel_size=3,
        dropout_frac=0.0,
        mean_initializer='glorot_uniform',
        log_var_initializer='zeros'
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

    mean = tf.keras.layers.Dense(latent_dim, kernel_initializer=mean_initializer)(x)
    log_var = tf.keras.layers.Dense(latent_dim, bias_initializer=log_var_initializer)(x)
    return tf.keras.models.Model(inputs, [mean, log_var])


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

    initial_pixels = int(np.ceil(pixels / 2 ** depth))
    x = tf.keras.layers.Dense(width * initial_pixels ** 2)(inputs)
    x = tf.keras.layers.Reshape((initial_pixels, initial_pixels, width))(x)
    for _ in range(depth):
        width = int(width / 2)
        x = tf.keras.layers.Conv2DTranspose(width, kernel_size=kernel_size, strides=2, padding='same')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_frac)(x)
    logits = tf.keras.layers.Conv2D(channels, kernel_size=kernel_size, padding='same')(x)

    crop_pixels = int((initial_pixels * 2 ** depth - pixels) / 2)
    if crop_pixels > 0:
        logits = tf.keras.layers.Cropping2D(((crop_pixels, crop_pixels), (crop_pixels, crop_pixels)))(logits)

    pixels = tf.keras.layers.Activation('sigmoid')(logits)
    return tf.keras.models.Model(inputs, [logits, pixels])


class AdversarialVAE(tf.keras.Model):
    """A keras model for training a VAE with a discriminator loss on the latent embedding.
    """
    def __init__(
            self,
            encoder,
            decoder,
            discriminator,
            random_generator,
            beta=1.0,
            gp_coef=10.0,
            gp_mode='interpolate'
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.random_generator = random_generator
        self.beta = beta
        self.gp_coef = gp_coef
        self.gp_mode = gp_mode

    def call(self, x, training=True):
        z_ = self.encode(x, training=training)
        s_ = self.discriminator(z_, training=training)
        logits_, pixels_ = self.decoder(z_, training=training)
        return z_, s_, logits_, pixels_

    @tf.function
    def compute_discriminator_loss(self, x):
        z_ = self.encode(x, training=True)
        z = self.random_generator.normal(tf.shape(z_))
        s_ = self.discriminator(z_, training=True)
        s = self.discriminator(z, training=True)

        loss = -tf.math.log(tf.nn.sigmoid(s))
        loss += -tf.math.log(tf.nn.sigmoid(-s_))

        if self.gp_coef > 0:
            gp = calculate_gradient_penalty(self.discriminator, self.random_generator, z, z_, True, self.gp_mode)
            loss += self.gp_coef * gp

        loss = tf.reduce_sum(loss)
        loss += tf.reduce_sum(self.discriminator.losses)
        return loss

    @tf.function
    def compute_generator_loss(self, x):
        z_, s_, logits_, pixels_ = self(x, training=True)

        recon_loss = continuous_bernoulli_loss(x, logits_)
        latent_loss = -tf.math.log(tf.nn.sigmoid(s_))

        loss = tf.reduce_sum(recon_loss)
        loss += self.beta * tf.reduce_sum(latent_loss)
        loss += tf.reduce_sum(self.encoder.losses)
        loss += tf.reduce_sum(self.decoder.losses)
        return loss

    def update_discriminator(self, loss, tape, optimizer):
        update_model(self.discriminator, loss, tape, optimizer)

    def update_generator(self, loss, tape, optimizer):
        trainable_variables = self.encoder.trainable_variables[:]
        trainable_variables += self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    @tf.function
    def encode(self, x, training=False, deterministic=False):
        mean, log_var = self.encoder(x, training=training)
        z_ = mean
        if not deterministic:
            noise = self.random_generator.normal(tf.shape(mean))
            z_ += noise * tf.math.exp(0.5 * log_var)
        return z_
