# Stardist

Training and prediction scripts for [stardist](https://github.com/mpicbg-csbd/stardist) models in 2d and 3d.
These scripts were adapted from the [stardist example notebooks](https://github.com/mpicbg-csbd/stardist/tree/master/examples).

## Installation

In order to install the software, you need miniconda. If you have not installed it yet, please download and install it [following the online instructions](https://docs.conda.io/en/latest/miniconda.html).
Once you have miniconda installed, make sure it is activated. Then you can install the requirements and activate it via:
```
conda env create -f environment_gpu.yaml
conda activate stardist-gpu
```
or, if you don't have a gpu available, via
```
conda env create -f environment_cpu.yaml
conda activate stardist-cpu
```

Finally, install the scripts to the environment via running
```
pip install -e .
```
in this folder.


Note that on the EMBL cluster, you need to make sure to use the correct OpenMPI version: run
```
module load OpenMPI/3.1.4-GCC-7.3.0-2.30
```
**BEFORE** the installation steps.


## Running the scripts

You can run the following scripts to train or predict a stardist model:
```
CUDA_VISIBLE_DEVICES=0 train_stardist_2d /path/to/data /path/to/model
```
```
CUDA_VISIBLE_DEVICES=0 predict_stardist_2d /path/to/data /path/to/model
```

The `CUDA_VISIBLE_DEVICES=0` part determines which gpu is used. If you have a machine with multiple GPUs and don't want to
use the first one, you need to change the `0` to the id of the GPU you want to use.

In order to run these scripts on the embl via slurm, you can use the `submit_slurm` script from `deep_cell.utils`, e.g.
```
submit_slurm train_stardist_2d /path/to/data /path/to/model
```
