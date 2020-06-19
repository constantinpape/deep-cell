import argparse
import json
import os
from glob import glob

import imageio
import numpy as np

from csbdeep.utils import normalize
from stardist import (calculate_extents, fill_label_holes,
                      gputools_available, Rays_GoldenSpiral)
from stardist.models import Config3D, StarDist3D


def check_training_data(train_images, train_labels):
    train_names = [os.path.split(train_im)[1] for train_im in train_images]
    label_names = [os.path.split(label_im)[1] for label_im in train_labels]
    assert len(train_names) == len(label_names), "Number of training images and label masks does not match"
    assert len(set(train_names) - set(label_names)) == 0, "Image names and label mask names do not match"


def check_training_images(train_images, train_labels):
    assert all(im.ndim == 3 for im in train_images), "Inconsistent image dimensions"
    assert all(im.ndim == 3 for im in train_labels), "Inconsistent label dimensions"
    assert all(label.shape == im.shape
               for label, im in zip(train_labels, train_images)), "Incosistent shapes of images and labels"


def load_training_data(root, image_folder, labels_folder, ext):

    # get the image and label mask paths and validate them
    image_pattern = os.path.join(root, image_folder, f'*{ext}')
    print("Looking for images with the pattern", image_pattern)
    train_images = glob(image_pattern)
    assert len(train_images) > 0, "Did not find any images"
    train_images.sort()

    label_pattern = os.path.join(root, labels_folder, f'*{ext}')
    print("Looking for labels with the pattern", image_pattern)
    train_labels = glob(label_pattern)
    assert len(train_labels) > 0, "Did not find any labels"
    train_labels.sort()

    check_training_data(train_images, train_labels)

    # normalization parameters: lower and upper percentile used for image normalization
    # maybe these should be exposed
    lower_percentile = 1
    upper_percentile = 99.8
    ax_norm = (0, 1, 2)

    train_images = [imageio.volread(im) for im in train_images]
    train_labels = [imageio.volread(im) for im in train_labels]
    check_training_images(train_images, train_labels)
    train_images = [normalize(im, lower_percentile, upper_percentile, axis=ax_norm)
                    for im in train_images]
    train_labels = [fill_label_holes(im) for im in train_labels]

    return train_images, train_labels


def make_train_val_split(train_images, train_labels, validation_fraction):
    n_samples = len(train_images)

    # we do train/val split with a fixed seed in order to be reproducible
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)
    n_val = max(1, int(validation_fraction * n_samples))
    train_indices, val_indices = indices[:-n_val], indices[-n_val:]
    x_train, y_train = [train_images[i] for i in train_indices], [train_labels[i] for i in train_indices]
    x_val, y_val = [train_images[i] for i in val_indices], [train_labels[i] for i in val_indices]

    return x_train, y_train, x_val, y_val


# TODO add more augmentations and refactor this so it can be used elsewhere
def random_flips_and_rotations(x, y):

    # first, rotate randomly
    axes = tuple(range(x.ndim))
    permute = np.random.permutation(axes)
    x, y = x.transpose(permute), y.transpose(permute)

    # second, flip randomly
    for ax in axes:
        if np.random.rand() > .5:
            x, y = np.flip(x, axis=ax), np.flip(y, axis=ax)

    return x, y


# multiplicative and additive random noise
def random_uniform_noise(x):
    return x * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)


def augmenter(x, y):
    x, y = random_flips_and_rotations(x, y)
    x = random_uniform_noise(x)
    return x, y


# we leave n_rays at the default of 32, but may want to expose this as well
def train_model(x_train, y_train,
                x_val, y_val,
                save_path,
                patch_size,
                anisotropy,
                n_rays=96):

    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)
    # make the model config
    # copied from the stardist training notebook, this is a very weird line ...
    use_gpu = False and gputools_available()
    # predict on subsampled image for increased efficiency
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)
    config = Config3D(
        rays=rays,
        grid=grid,
        use_gpu=use_gpu,
        n_channel_in=1,
        train_patch_size=patch_size,
        anisotropy=anisotropy
    )

    if use_gpu:
        print("Using a GPU for training")
        # limit gpu memory
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(0.8)
    else:
        print("GPU not found, using the CPU for training")

    save_root, save_name = os.path.split(save_path)
    os.makedirs(save_root, exist_ok=True)
    model = StarDist3D(config, name=save_name, basedir=save_root)

    model.train(x_train, y_train, validation_data=(x_val, y_val), augmenter=augmenter)
    optimal_parameters = model.optimize_thresholds(x_val, y_val)
    return optimal_parameters


def compute_anisotropy_from_data(data):
    extents = calculate_extents(data)
    anisotropy = tuple(np.max(extents) / extents)
    return anisotropy


def train_stardist_model(root, model_save_path, image_folder, labels_folder, ext,
                         validation_fraction, patch_size, anisotropy):
    print("Loading training data")
    train_images, train_labels = load_training_data(root, image_folder, labels_folder, ext)
    print("Found", len(train_images), "images and label masks for training")

    x_train, y_train, x_val, y_val = make_train_val_split(train_images, train_labels,
                                                          validation_fraction)
    if anisotropy is None:
        anisotropy = compute_anisotropy_from_data(y_train)
        print("Anisotropy factor was computed from data:", anisotropy)

    print("Made train validation split with validation fraction", validation_fraction, "resulting in")
    print(len(x_train), "training images")
    print(len(y_train), "validation images")

    print("Start model training ...")
    print("You can connect to the tensorboard by typing 'tensorboaed --logdir=.' in the folder where the training runs")
    optimal_parameters = train_model(x_train, y_train, x_val, y_val, model_save_path, patch_size, anisotropy)
    print("The model has been trained and was saved to", model_save_path)
    print("The following optimal parameters were found:", optimal_parameters)


# use configarparse?
# TODO set batch size
# TODO enable fine-tuning on pre-trained
# TODO enable excluding images by name
def main():
    parser = argparse.ArgumentParser(description="Train a 3d stardist model")
    parser.add_argument('root', type=str, help="Root folder with folders for the training images and labels.")
    parser.add_argument('model_save_path', type=str, help="Where to save the model.")
    parser.add_argument('--image_folder', type=str, default='images',
                        help="Name of the folder with the training images, default: images.")
    parser.add_argument('--labels_folder', type=str, default='labels',
                        help="Name of the folder with the training labels, default: labels.")
    parser.add_argument('--ext', type=str, default='.tif', help="Image file extension, default: .tif")
    parser.add_argument('--validation_fraction', type=float, default=.1,
                        help="The fraction of available data that is used for validation, default: .1")
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128],
                        help="Size of the image patches used to train the network, default: 128, 128, 128")
    aniso_help = """Anisotropy factor, needs to be passed as json encoded list, e.g. \"[.05,0.5,0.5]\".
                    If not given, will be computed from the dimensions of the input data, default: None"""
    parser.add_argument('--anisotropy', type=str, default=None,
                        help=aniso_help)

    args = parser.parse_args()
    anisotropy = args.anisotropy
    if anisotropy is not None:
        anisotropy = json.loads(anisotropy)

    train_stardist_model(args.root, args.model_save_path,
                         args.image_folder, args.labels_folder,
                         args.ext, args.validation_fraction,
                         tuple(args.patch_size), anisotropy)


if __name__ == '__main__':
    main()
