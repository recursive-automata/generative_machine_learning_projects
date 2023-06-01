# -*- coding: utf-8 -*-

import tensorflow as tf

PI = 3.14159


class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate, warmup_steps=4000, decay_steps=20000):
        super().__init__()
        self.rate = tf.cast(rate, dtype=tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
        self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)

    @tf.function
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        step = tf.math.mod(step, self.warmup_steps + self.decay_steps)

        arg1 = (1 + step) / (1 + self.warmup_steps)

        arg2 = (step - self.warmup_steps) / self.decay_steps
        arg2 = tf.math.cos(PI * arg2)
        arg2 = 0.5 * (1 + arg2)

        return self.rate * tf.where(step <= self.warmup_steps, arg1, arg2)