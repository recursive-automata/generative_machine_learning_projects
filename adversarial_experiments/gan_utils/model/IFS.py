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


class IFSLayer(tf.keras.layers.Layer):
    """A keras layer representing an Iterated Function System.

    Iterated Function Systems (IFS) is used to model complex sets in metric spaces,
    including certain classes of fractal.
    An IFS consists of a collection of affine transformations with Lipschitz coefficients less than 1.

    To apply an IFS to a point, randomly sample one of the IFS's transformations
    (or chose a transformation according to a given function's index)
    and apply it to the point.
    If this process is repeated ad infinitum,
    the point set output approaches a unique limit under the Hausdorff metric.
    """
    def __init__(self, n_dim, n_functions, max_norm=0.99):
        super(IFSLayer, self).__init__()
        self.n_dim = n_dim
        self.n_functions = n_functions
        self.functions = tuple(
            tf.keras.layers.Dense(
                n_dim,
                kernel_constraint=tf.keras.constraints.MaxNorm(max_norm)
            )
            for _ in range(n_functions)
        )

    @tf.function
    def call(self, args):
        """Applies the Iterated Function System.

        :param args: `(x, function_indices)`, where x has type float and shape `(n_obs, n_dims)`
        and function_indices has type int32 and shape `(n_obs, n_applications)`.
        :return: a tensor of type float and shape `(n_obs, n_dims)`, the result of applying
        the transformations to x `n_applications` times.
        """
        x, function_indices = args
        n_applications = tf.shape(function_indices)[1]
        for i in range(n_applications):
            function_index = function_indices[:, i]
            mask = tf.one_hot(function_index, self.n_functions)
            mask = mask[:, :, None]
            x = [f(x)[:, None, :] for f in self.functions]
            x = tf.concat(x, axis=1)
            x = tf.reduce_sum(mask * x, axis=1)
        return x


def generate_random_function_indices(n_functions, batch_size, n_applications, random_generator=None):
    """Generate random sequences of functions to apply when calling IFSLayers.

    :param n_functions: How many functions are in the IFS
    :param batch_size: How many observations per batch
    :param n_applications: How many function applications
    :param random_generator: A tf random generator object (optional)
    :return: a tensor of type int32 and shape `(batch_size, n_applications)`
    whose entries are sampled uniformly from {0, ...,  n_functions - 1}
    """
    random_generator = tf.random if random_generator is None else random_generator
    return tf.cast(
        tf.math.floor(n_functions * random_generator.uniform([batch_size, n_applications])),
        'int32'
    )
