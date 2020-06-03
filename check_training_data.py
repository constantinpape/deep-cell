import argparse
import os
from glob import glob

import imageio
import napari


def check_training_data(root, image_folder, labels_folder, ext):
    image_folder = os.path.join(root, image_folder)
    assert os.path.exists(image_folder), f"Could not find {image_folder}"
    labels_folder = os.path.join(root, labels_folder)
    assert os.path.exists(labels_folder), f"Could not find {labels_folder}"

    files = glob(os.path.join(image_folder, f"*{ext}"))
    files.sort()

    for ff in files:
        try:
            im = imageio.imread(ff)
            name = os.path.split(ff)[1]
        except Exception as e:
            print(f"Could not open {ff}")
            print(f"Failed with {e}")
            continue
        try:
            label_file = os.path.join(labels_folder, name)
            labels = imageio.imread(label_file)
        except Exception as e:
            print(f"Could not open {label_file}")
            print(f"Failed with {e}")
            continue

        with napari.gui_qt():
            viewer = napari.Viewer(title=name)
            viewer.add_image(im)
            viewer.add_labels(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--image_folder', type=str, default='images')
    parser.add_argument('--labels_folder', type=str, default='labels')
    parser.add_argument('--ext', type=str, default='.tif')

    args = parser.parse_args()
    check_training_data(args.root, args.image_folder, args.labels_folder, args.ext)
