import argparse
import os
from glob import glob

import imageio
import numpy as np

from csbdeep.utils import normalize
from stardist import fill_label_holes, gputools_available
from stardist.models import Config2D, StarDist2D


def check_training_data(train_images, train_labels):
    train_names = [os.path.split(train_im)[1] for train_im in train_images]
    label_names = [os.path.split(label_im)[1] for label_im in train_labels]
    assert len(train_names) == len(label_names), "Number of training images and label masks does not match"
    assert len(set(train_names) - set(label_names)) == 0, "Image names and label mask names do not match"


def check_training_images(train_images, train_labels):

    ndim = train_images[0].ndim
    assert all(im.ndim == ndim for im in train_images), "Inconsistent image dimensions"
    assert all(im.ndim == 2 for im in train_labels), "Inconsistent label dimensions"

    def get_n_channels(im):
        return 1 if im.ndim == 2 else im.shape[-1]

    def get_im_shape(im):
        return im.shape if im.ndim == 2 else im.shape[:-1]

    n_channels = get_n_channels(train_images[0])
    assert all(get_n_channels(im) == n_channels for im in train_images), "Inconsistent number of image channels"
    assert all(label.shape == get_im_shape(im)
               for label, im in zip(train_labels, train_images)), "Incosistent shapes of images and labels"

    return n_channels


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

    # load the images, check tham and preprocess the data
    train_images = [imageio.imread(im) for im in train_images]
    train_labels = [imageio.imread(im) for im in train_labels]
    n_channels = check_training_images(train_images, train_labels)
    train_images = [normalize(im, lower_percentile, upper_percentile) for im in train_images]
    train_labels = [fill_label_holes(im) for im in train_labels]

    return train_images, train_labels, n_channels


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
def train_model(x_train, y_train, x_val, y_val, save_path,
                n_channels, patch_size, n_rays=32):

    # make the model config
    # copied from the stardist training notebook, this is a very weird line ...
    use_gpu = False and gputools_available()
    # predict on subsampled image for increased efficiency
    grid = (2, 2)
    config = Config2D(
        n_rays=n_rays,
        grid=grid,
        use_gpu=use_gpu,
        n_channel_in=n_channels,
        train_patch_size=patch_size
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
    model = StarDist2D(config, name=save_name, basedir=save_root)

    model.train(x_train, y_train, validation_data=(x_val, y_val), augmenter=augmenter)
    optimal_parameters = model.optimize_threshold(x_val, y_val)
    return optimal_parameters


def train_stardist_model(root, model_save_path, image_folder, labels_folder, ext,
                         validation_fraction, patch_size):
    print("Loading training data")
    train_images, train_labels, n_channels = load_training_data(root, image_folder, labels_folder, ext)
    print("Found", len(train_images), "images and label masks for training")

    x_train, y_train, x_val, y_val = make_train_val_split(train_images, train_labels,
                                                          validation_fraction)
    print("Made train validation split with validation fraction", validation_fraction, "resulting in")
    print(len(x_train), "training images")
    print(len(y_train), "validation images")

    print("Start model training ...")
    print("You can connect to the tensorboard by typing 'tensorboaed --logdir=.' in the folder where the training runs")
    optimal_parameters = train_model(x_train, y_train, x_val, y_val, model_save_path,
                                     n_channels, patch_size)
    print("The mode has been trained and was saved to", model_save_path)
    print("The following optimal parameters were found:", optimal_parameters)


# use configarparse?
# TODO enable fine-tuning on pre-trained
# TODO enable excluding images by name
def main():
    parser = argparse.ArgumentParser(description="Train a 2d stardist model")
    parser.add_argument('root', type=str, help="Root folder with folders for the training images and labels.")
    parser.add_argument('model_save_path', type=str, help="Where to save the model.")
    parser.add_argument('--image_folder', type=str, default='images',
                        help="Name of the folder with the training images, default: images.")
    parser.add_argument('--labels_folder', type=str, default='labels',
                        help="Name of the folder with the training labels, default: labels.")
    parser.add_argument('--ext', type=str, default='.tif', help="Image file extension, default: .tif")
    parser.add_argument('--validation_fraction', type=float, default=.1,
                        help="The fraction of available data that is used for validation, default: .1")
    parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256],
                        help="Size of the image patches used to train the network, default: 256, 256")

    args = parser.parse_args()
    train_stardist_model(args.root, args.model_save_path,
                         args.image_folder, args.labels_folder,
                         args.ext, args.validation_fraction,
                         tuple(args.patch_size))


if __name__ == '__main__':
    main()
