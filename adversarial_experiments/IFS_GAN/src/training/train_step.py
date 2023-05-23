# -*- coding: utf-8 -*-

import tensorflow as tf


def make_train_step(ifs_gan, generator_optimizer, discriminator_optimizer, discriminator_steps=1):
    def _train_step(x, codes, step_number):
        with tf.GradientTape() as tape:
            loss = ifs_gan.compute_discriminator_loss(x, codes)
        ifs_gan.update_discriminator(loss, tape, generator_optimizer)

        # only train the generator once every discriminator_steps
        if (step_number + 1) % discriminator_steps == 0:
            with tf.GradientTape() as tape:
                loss = ifs_gan.compute_generator_loss(x, codes)
            ifs_gan.update_generator(loss, tape, discriminator_optimizer)
    return _train_step
