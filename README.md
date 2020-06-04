# Deep Cell

Training and inference scripts for deep learning tools for cell segmentation in microscopy images.

Available tools:
- [stardist](https://github.com/mpicbg-csbd/stardist): stardist for convex object segmentation
- utils: functionality for visualisation and job submission on compute clusters


## Data Layout

The goal of this small package is to provide an easy way to train different tools via the command line from the cell data layout.
In order to use it, you will need training data images and labels in the following layout:
```
root-folder/
  images/
  labels/
```
The folder `images` contains the training image data and the labels the training label data.
The corresponding images and labels **must have exactly the same name**.
The data should be stored in tif format. For multi-channel images, we assume that they are stored channel-first, i.e. in cyx order.
