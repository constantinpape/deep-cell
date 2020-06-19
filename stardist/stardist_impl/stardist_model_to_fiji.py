import argparse
import os
from stardist.models import StarDist2D


def stardist_model_to_fiji(model_path, model=None):

    if model is None:
        save_root, save_name = os.path.split(model_path)
        model = StarDist2D(None, name=save_name, basedir=save_root)

    fiji_save_path = os.path.join(model_path, 'TF_SavedModel.zip')
    print("Saving model for fiji", fiji_save_path)
    model.export_TF()


def main():
    parser = argparse.ArgumentParser(description="Save a stardist model for fiji")
    parser.add_argument('model_path', type=str, help="Where the model is saved.")

    args = parser.parse_args()
    stardist_model_to_fiji(args.model_path)


if __name__ == '__main__':
    main()
