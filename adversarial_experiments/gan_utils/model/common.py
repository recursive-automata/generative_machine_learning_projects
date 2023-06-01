# -*- coding: utf-8 -*-


def update_model(model, loss, tape, optimizer):
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
