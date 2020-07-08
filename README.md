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


## Setting up conda

The software dependencies of this repository are installed via conda.
If you don't have conda installed yet, check out the [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
Note that there are two different versions of conda:
Anaconda, which comes with the conda package manager and a complete environment with popular python packages as well as miniconda, which only contains the conda package manager.
For our purposes, it is sufficient to install miniconda (but using anaconda does not hurt if you need it for some other purpose).

### Setting up a multi-user conda installation

In order to set up a shared conda installation  for multiple users on linux infrastrcture, follow these steps:

1. Download the latest version of miniconda: https://docs.conda.io/en/latest/miniconda.html. 
2. Open a terminal and cd to the location where you have downloaded the installation script
3. Set executable permissions: `chmod +x Miniconda3-latest-Linux-x86_64.sh`
4. Exectute the installation script: `./Miniconda3-latest-Linux-x86_64.sh`
5. Agree to the license
6. Choose the installation directory and start the installation. Make sure that all users have read and execute permissions for the installation location.
7. After the installation is finished, choose `no` when asked `Do you wish to initialize Miniconda3 by running conda init?`.
8. This will end the installation and print out a few lines explaining how to activate the conda base environment.
9. Copy the line `eval "$(/path/to/miniconda3/bin/conda shell.YOUR_SHELL_NAME hook)"`and paste it into a file `init_conda.sh`.
10. Replace `YOUR_SHELL_NAME` with `bash` (assuming you and your users are using a bash shell; for zshell, replace with `zsh`, etc.).

Now, the conda base environment can be activated via running `source init_conda.sh`. 
Use it to set up the environments to make the applications available to your users.
Important: in order to install version conflicts use separate environments for different application and don't install them to the base environment!

In order for users to activate one of these environments, they will need to first activate the base environment and then the desired env:
```shell
source init_conda.sh
conda activate <ENV_NAME>
```

### Advanced: setting up a conda environment for multiple maintainers and users

TODO
