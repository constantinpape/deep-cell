import argparse
import os
from glob import glob

import imageio
from tqdm import tqdm

from csbdeep.utils import normalize
from stardist.models import StarDist2D


def get_image_files(root, image_folder, ext):
    # get the image and label mask paths and validate them
    image_pattern = os.path.join(root, image_folder, f'*{ext}')
    print("Looking for images with the pattern", image_pattern)
    images = glob(image_pattern)
    assert len(images) > 0, "Did not find any images"
    images.sort()

    return images


# could be done more efficiently, see
# https://github.com/hci-unihd/batchlib/blob/master/batchlib/segmentation/stardist_prediction.py
def run_prediction(image_files, model_path, root, prediction_folder, multichannel):

    # load the model
    model_root, model_name = os.path.split(model_path.rstrip('/'))
    model = StarDist2D(None, name=model_name, basedir=model_root)

    res_folder = os.path.join(root, prediction_folder)
    os.makedirs(res_folder, exist_ok=True)

    # normalization parameters: lower and upper percentile used for image normalization
    # maybe these should be exposed
    lower_percentile = 1
    upper_percentile = 99.8
    ax_norm = (0, 1)  # independent normalization for multichannel images

    for im_file in tqdm(image_files, desc="run stardist prediction"):
        if multichannel:
            im = imageio.volread(im_file).transpose((1, 2, 0))
        else:
            im = imageio.imread(im_file)
        im = normalize(im, lower_percentile, upper_percentile, axis=ax_norm)
        pred, _ = model.predict_instances(im)

        im_name = os.path.split(im_file)[1]
        save_path = os.path.join(res_folder, im_name)
        imageio.imsave(save_path, pred)


def predict_stardist(root, model_path, image_folder, prediction_folder, ext, multichannel):
    print("Loading images")
    image_files = get_image_files(root, image_folder, ext)
    print("Found", len(image_files), "images for prediction")

    print("Start prediction ...")
    run_prediction(image_files, model_path, root, prediction_folder, multichannel)
    print("Finished prediction")


def main():
    parser = argparse.ArgumentParser(description="Predict new images with a stardist model")
    parser.add_argument('root', type=str, help="Root folder with image data.")
    parser.add_argument('model_path', type=str, help="Where the model is saved.")
    parser.add_argument('--image_folder', type=str, default='images',
                        help="Name of the folder with the training images, default: images.")
    parser.add_argument('--prediction_folder', type=str, default='predictions',
                        help="Name of the folder where the predictions should be stored, default: predictions.")
    parser.add_argument('--ext', type=str, default='.tif', help="Image file extension, default: .tif")
    parser.add_argument('--multichannel', type=int, default=0, help="Do we have multichannel images? Default: 0")

    args = parser.parse_args()
    predict_stardist(args.root, args.model_path, args.image_folder, args.prediction_folder,
                     args.ext, args.multichannel)


if __name__ == '__main__':
    main()
