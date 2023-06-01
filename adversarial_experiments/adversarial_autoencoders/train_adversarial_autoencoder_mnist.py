# -*- coding: utf-8 -*-
#!/usr/bin/python3

"""
Train an IFS-GAN to learn the Koch curve from programmatically generated sample points.
"""

import argparse
import tensorflow as tf

from model.discriminator import build_mlp
from model.vae import AdversarialVAE
from model.vae import build_convnet_decoder
from model.vae import build_convnet_encoder
from training.save_image import make_video
from training.save_image import scatter
from training.schedule import CosineAnnealingSchedule
from training.train_step import make_train_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--warmup-epochs", default=15, type=int, help='How many epochs to linearly ramp up to the full learning rate')
    parser.add_argument("--decay-epochs", default=45, type=int, help='How many epochs to cosine decay down to zero learning rate')
    parser.add_argument("--lr", default=3e-4, type=float, help='Learning rate for the IFS')
    parser.add_argument("--disc-lr", default=3e-4, type=float, help='Learning rate for the discriminator')
    parser.add_argument("--width", default=512, type=int, help='Autoencoder width')
    parser.add_argument("--disc-width", default=512, type=int, help='Discriminator neurons per layer')
    parser.add_argument("--disc-depth", default=4, type=int, help='Discriminator number of layers')
    parser.add_argument("--encoder-dropout", default=0.2, type=float, help='Encoder last-layer dropout coefficient')
    parser.add_argument("--decoder-dropout", default=0.2, type=float, help='Decoder last-layer dropout coefficient')
    parser.add_argument("--disc-dropout", default=0.5, type=float, help='Discriminator last-layer dropout coefficient')
    parser.add_argument("--beta", default=3.0, type=float, help='Weight for the latent term in the loss')
    parser.add_argument("--gp-coef", default=10.0, type=float, help='Discriminator gradient penalty coefficient')
    parser.add_argument("--disc-steps", default=1, type=int, help='How many discriminator steps per generator steps')
    parser.add_argument("--image-file-prefix", default='./images/mnist_latent')
    parser.add_argument("--model-save-file", default='./models/mnist_adversarial_autoencoder')
    parser.add_argument("--fps", default=6, type=int, help='Frames per second for video output')
    parser.add_argument("--tf-random-seed", default=42, type=int, help='Optional random seed')
    return parser.parse_args()


def _preprocess_mnist(images):
    images = images.reshape((images.shape[0], 28, 28, 1))
    images = images / 255.
    return images.astype('float32')


def make_dataset(images, batch_size):
    dataset = _preprocess_mnist(images)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    size = len(images)
    dataset = dataset.shuffle(size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def run_main_loop(dataset, train_step, model, epochs, x0, y0, image_file_prefix, xlim, ylim):
    for epoch in range(epochs):
        for n, x in enumerate(dataset):
            train_step(x, n)

        z = model.encode(x0, training=False, deterministic=True).numpy()
        filepath = f'{image_file_prefix}_{epoch:04d}.png'
        scatter(z, filepath, xlim, ylim, axis=None, figsize=(12, 12), s=0.5, c=y0)


def main():
    args = parse_args()

    latent_dim = 2
    xlim = (-4.0, 4.0)
    ylim = (-4.0, 4.0)
    beta_1 = 0.5
    beta_2 = 0.9

    if args.tf_random_seed is not None:
        tf.random.set_seed(args.tf_random_seed)

    if args.tf_random_seed is not None:
        random_generator = tf.random.Generator.from_seed(args.tf_random_seed)
    else:
        random_generator = tf.random.Generator.from_non_deterministic_state()

    mnist = tf.keras.datasets.mnist
    (train_images, _), (test_images, test_labels) = mnist.load_data()

    train_dataset = make_dataset(train_images, args.batch_size)
    x0 = _preprocess_mnist(test_images)
    y0 = test_labels

    # initialize so the latents start near the origin
    s0 = 0.3
    mean_initializer = tf.keras.initializers.Orthogonal(gain=s0)
    log_var_initializer = tf.keras.initializers.Constant(float(2 * tf.math.log(s0)))
    encoder = build_convnet_encoder(
        28,
        1,
        latent_dim,
        args.width,
        dropout_frac=args.encoder_dropout,
        mean_initializer=mean_initializer,
        log_var_initializer=log_var_initializer
    )
    decoder = build_convnet_decoder(28, 1, latent_dim, args.width, dropout_frac=args.decoder_dropout)
    discriminator = build_mlp(latent_dim, args.disc_width, args.disc_depth, args.disc_dropout)
    model = AdversarialVAE(encoder, decoder, discriminator, random_generator, beta=args.beta, gp_coef=args.gp_coef)

    steps_per_epoch = int(len(train_images) / args.batch_size)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch

    schedule = CosineAnnealingSchedule(args.lr, warmup_steps, decay_steps)
    optimizer = tf.keras.optimizers.Adam(schedule, beta_1=beta_1, beta_2=beta_2)
    disc_schedule = CosineAnnealingSchedule(args.disc_lr, warmup_steps, decay_steps)
    disc_optimizer = tf.keras.optimizers.Adam(disc_schedule, beta_1=beta_1, beta_2=beta_2)
    train_step = make_train_step(model, optimizer, disc_optimizer, args.disc_steps)

    run_main_loop(train_dataset, train_step, model, args.epochs, x0, y0, args.image_file_prefix, xlim, ylim)
    model.save(args.model_save_file)
    make_video(args.image_file_prefix, fps=args.fps)


if __name__ == '__main__':
    main()
