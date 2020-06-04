import argparse
import os
from glob import glob

import imageio
import napari


def view_data(root, image_folder, labels_folder, prediction_folder,
              ext, prediction_is_labels):
    image_folder = os.path.join(root, image_folder)
    assert os.path.exists(image_folder), f"Could not find {image_folder}"

    if labels_folder is not None:
        labels_folder = os.path.join(root, labels_folder)
        assert os.path.exists(labels_folder), f"Could not find {labels_folder}"

    if prediction_folder is not None:
        prediction_folder = os.path.join(root, prediction_folder)
        assert os.path.exists(prediction_folder), f"Could not find {prediction_folder}"

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

    # TODO instead of looping over images load them in napari with selection gui
    for ff in files:
        im, name = _load(ff)

        if im is None:
            continue

        if labels_folder is not None:
            label_file = os.path.join(labels_folder, name)
            labels, _ = _load(label_file)
        else:
            labels = None

        if prediction_folder is not None:
            pred_file = os.path.join(prediction_folder, name)
            prediction, _ = _load(pred_file)
        else:
            prediction = None

        with napari.gui_qt():
            viewer = napari.Viewer(title=name)
            viewer.add_image(im)

            if labels is not None:
                viewer.add_labels(labels)

            if prediction is not None:
                if prediction_is_labels:
                    viewer.add_labels(prediction)
                else:
                    viewer.add_image(prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--image_folder', type=str, default='images')
    parser.add_argument('--labels_folder', type=str, default=None)
    parser.add_argument('--prediction_folder', type=str, default=None)
    parser.add_argument('--prediction_is_labels', type=int, default=1)
    parser.add_argument('--ext', type=str, default='.tif')

    args = parser.parse_args()
    view_data(args.root, args.image_folder, args.labels_folder, args.prediction_folder,
              args.ext, bool(args.prediction_is_labels))


if __name__ == '__main__':
    main()
