# -*- coding: utf-8 -*-
#!/usr/bin/python3

"""
Train an IFS-GAN to learn the Koch curve from programmatically generated sample points.
"""

import argparse
import tensorflow as tf

from data.koch import make_koch_samples
from model.discriminator import build_mlp
from model.IFS import generate_random_function_indices
from model.IFS import IFSLayer
from model.IFS import IFSGAN
from training.hausdorff import calculate_hausdorff_suprema
from training.save_image import make_video
from training.save_image import scatter
from training.train_step import make_train_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=3e-4, type=float, help='Learning rate for the IFS')
    parser.add_argument("--disc-lr", default=3e-4, type=float, help='Learning rate for the discriminator')
    parser.add_argument("--width", default=64, type=int, help='Discriminator neurons per layer')
    parser.add_argument("--depth", default=3, type=int, help='Discriminator number of layers')
    parser.add_argument("--dropout", default=0.25, type=float, help='Discriminator last-layer dropout coefficient')
    parser.add_argument("--gp-coef", default=10.0, type=float, help='Discriminator gradient penalty coefficient')
    parser.add_argument("--disc-steps", default=1, type=int, help='How many discriminator steps per generator steps')
    parser.add_argument("--n_applications", default=6, type=int, help='How many IFS iterations to apply during training')
    parser.add_argument("--max_norm", default=0.334, type=float, help='Max Lipschitz coefficient for the IFS\'s transformations')
    parser.add_argument("--image-file-prefix", default='./images/koch')
    parser.add_argument("--model-save-file", default='./models/koch_ifs_gan')
    parser.add_argument("--fps", default=6, type=int, help='Frames per second for video output')
    parser.add_argument("--tf-random-seed", default=42, type=int, help='Optional random seed')
    return parser.parse_args()


def make_dataset(sample_depth, batch_size, n_functions, n_applications, random_generator):
    koch = make_koch_samples(sample_depth)
    dataset = tf.data.Dataset.from_tensor_slices(koch.astype('float32'))
    dataset = dataset.shuffle(len(koch))

    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.map(lambda x: (
        x,
        generate_random_function_indices(n_functions, batch_size, n_applications, random_generator)
    ))
    return dataset


def run_main_loop(dataset, train_step, ifs_gan, epochs, x0, i0, image_file_prefix, xlim, ylim, logger=None):
    for epoch in range(epochs):
        for n, x in enumerate(dataset):
            train_step(x, n)

        x_ = ifs_gan.encode(x0, i0)

        if logger is not None:
            d, d_ = calculate_hausdorff_suprema(x, x_)
            logger.info(f'Epoch {epoch}, samplewise suprema distances {d:04f} and {d_:04f}')

        filepath = f'{image_file_prefix}_{epoch:04d}.png'
        scatter(x_.numpy(), filepath, xlim, ylim, s=0.5)


def main():
    args = parse_args()

    sample_depth = 8
    n_functions = 4
    n_dim = 2
    xlim = (-1.2, 1.2)
    ylim = (-0.3, 0.5)

    if args.tf_random_seed is not None:
        tf.random.set_seed(args.tf_random_seed)

    if args.tf_random_seed is not None:
        random_generator = tf.random.Generator.from_seed(args.tf_random_seed)
    else:
        random_generator = tf.random.Generator.from_non_deterministic_state()

    dataset = make_dataset(sample_depth, args.batch_size, n_functions, args.n_applications, random_generator)

    ifs_layer = IFSLayer(n_dim, n_functions, args.max_norm)
    discriminator = build_mlp(n_dim, args.width, args.depth, args.dropout)
    ifs_gan = IFSGAN(ifs_layer, discriminator, random_generator, gp_coef=args.gp_coef)

    beta_1 = 0.5
    beta_2 = 0.9
    optimizer = tf.keras.optimizers.Adam(args.lr, beta_1=beta_1, beta_2=beta_2)
    disc_optimizer = tf.keras.optimizers.Adam(args.disc_lr, beta_1=beta_1, beta_2=beta_2)
    train_step = make_train_step(ifs_gan, optimizer, disc_optimizer, args.disc_steps)

    # consistent samples for generated images
    x0 = random_generator.normal([args.batch_size, n_dim])
    i0 = generate_random_function_indices(n_functions, args.batch_size, args.n_applications, random_generator)
    run_main_loop(dataset, train_step, ifs_gan, args.epochs, x0, i0, args.image_file_prefix, xlim, ylim)

    ifs_gan.save(args.model_save_file)
    make_video(args.image_file_prefix, fps=args.fps)


if __name__ == '__main__':
    main()
