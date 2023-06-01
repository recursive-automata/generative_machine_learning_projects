# -*- coding: utf-8 -*-

import tensorflow as tf


def _continuous_bernoulli_norm_term(logits_):
    epsilon = 1e-8
    p = tf.nn.sigmoid(logits_)
    return (
        tf.math.log(epsilon + tf.math.abs(tf.math.atanh(1 - 2 * p)))
        - tf.math.log(epsilon + tf.math.abs(1 - 2 * p))
    )


def continuous_bernoulli_loss(pixels, logits_):
    """https://arxiv.org/abs/1907.06845
    """
    recon_term = tf.nn.sigmoid_cross_entropy_with_logits(labels=pixels, logits=logits_)
    norm_term = _continuous_bernoulli_norm_term(logits_)
    return recon_term + norm_term
