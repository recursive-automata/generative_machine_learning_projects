# Adversarial Autoencoders

Demonstration code for training disentangling autoencoders
using an adversarial latent loss.

Simon Schneider, May 2023

## Overview

This code replicates
[some old work on adversarial autoencoders](https://arxiv.org/pdf/1511.05644.pdf).
The core idea is to replace the KL-Divergence term in a Variational Autoencoder's loss function
with a learned adversarial loss function.
We also use an image reconstruction loss derived from
[the continuous Bernoulli distribution](https://arxiv.org/abs/1907.06845).

### To run this demo:

1. Prepare a python3 environment with numpy, matplotlib, and tensorflow.
2. Install the local `gan_utils` packages: `cd ..; python -m pip install .; cd IFS_GAN`.
3. Run the script: `train_adversarial_autoencoder_mnist.py`.
The script will create a video of the latent encodings of the test set
and write it to `images/mnist_latent_video.mp4`.
