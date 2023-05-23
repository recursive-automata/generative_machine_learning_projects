# -*- coding: utf-8 -*-

import tensorflow as tf

from .discriminator import calculate_gradient_penalty


def _update(model, loss, tape, optimizer):
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


class IFSGAN(tf.keras.Model):
    """A keras model for training an IFSLayer using an adversarial loss.
    """
    def __init__(self, ifs_layer, discriminator, random_generator, gp_coef=10.0):
        super().__init__()
        self.ifs_layer = ifs_layer
        self.discriminator = discriminator
        self.random_generator = random_generator
        self.gp_coef = gp_coef

    def call(self, inputs, training=True):
        x, codes = inputs
        noise = self.random_generator.normal(tf.shape(x))
        x_ = self.ifs_layer((noise, codes), training=training)
        p_ = self.discriminator(x_, training=training)
        return x_, p_

    @tf.function
    def compute_discriminator_loss(self, x, codes):
        x_, p_ = self([x, codes], training=True)
        p = self.discriminator(x, training=True)

        loss = -tf.math.log(tf.nn.sigmoid(p))
        loss += -tf.math.log(tf.nn.sigmoid(-p_))

        if self.gp_coef > 0:
            gp = calculate_gradient_penalty(self.discriminator, self.random_generator, x, x_)
            loss += self.gp_coef * gp

        loss = tf.reduce_sum(loss)
        loss += tf.reduce_sum(self.discriminator.losses)
        return loss

    def compute_generator_loss(self, x, codes):
        _, p_ = self([x, codes], training=True)
        loss = -tf.math.log(tf.nn.sigmoid(p_))

        loss = tf.reduce_sum(loss)
        loss += tf.reduce_sum(self.ifs_layer.losses)
        return loss

    def update_discriminator(self, loss, tape, optimizer):
        _update(self.discriminator, loss, tape, optimizer)

    def update_generator(self, loss, tape, optimizer):
        _update(self.ifs_layer, loss, tape, optimizer)

    def encode(self, x, codes):
        return self.ifs_layer((x, codes), training=False)

    def score(self, x):
        return self.discriminator(x, training=False)
