# -*- coding: utf-8 -*-

import tensorflow as tf


def make_train_step(gan, generator_optimizer, discriminator_optimizer, discriminator_steps=1):
    def _train_step(x, step_number):
        if (step_number + 1) % (discriminator_steps + 1) != 0:
            with tf.GradientTape() as tape:
                loss = gan.compute_discriminator_loss(x)
            gan.update_discriminator(loss, tape, generator_optimizer)

        if (step_number + 1) % (discriminator_steps + 1) == 0:
            with tf.GradientTape() as tape:
                loss = gan.compute_generator_loss(x)
            gan.update_generator(loss, tape, discriminator_optimizer)

    return _train_step
