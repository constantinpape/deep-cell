import argparse
import os
from glob import glob

import imageio
import napari


def check_training_data(root, image_folder, labels_folder, ext, prediction_folder,
                        prediction_is_labels):
    image_folder = os.path.join(root, image_folder)
    assert os.path.exists(image_folder), f"Could not find {image_folder}"
    labels_folder = os.path.join(root, labels_folder)
    assert os.path.exists(labels_folder), f"Could not find {labels_folder}"

    files = glob(os.path.join(image_folder, f"*{ext}"))
    files.sort()

    def _load(path):
        try:
            im = imageio.imread(path)
            name = os.path.split(path)[1]
        except Exception as e:
            print(f"Could not open {path}")
            print(f"Failed with {e}")
            im, name = None, None
        return im, name

    for ff in files:
        im, name = _load(ff)

        if im is None:
            continue

        label_file = os.path.join(labels_folder, name)
        labels, _ = _load(label_file)

        if prediction_folder is not None:
            pred_file = os.path.join(prediction_folder, name)
            pred, _ = _load(pred_file)

        with napari.gui_qt():
            viewer = napari.Viewer(title=name)
            viewer.add_image(im)
            viewer.add_labels(labels)
            if prediction_folder is not None:
                if prediction_is_labels:
                    viewer.add_labels(pred)
                else:
                    viewer.add_image(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--image_folder', type=str, default='images')
    parser.add_argument('--labels_folder', type=str, default='labels')
    parser.add_argument('--prediction_folder', type=str, default=None)
    parser.add_argument('--prediction_is_labels', type=int, default=1)
    parser.add_argument('--ext', type=str, default='.tif')

    args = parser.parse_args()
    check_training_data(args.root, args.image_folder, args.labels_folder, args.ext,
                        args.prediction_folder, bool(args.prediction_is_labels))
