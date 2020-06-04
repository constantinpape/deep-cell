# EMBL-Tools

Tools for data visualisation and to submit gpu jobs on the embl cluster.


## Installation

Activate your conda environment, e.g. from `../stardist/environment-gpu.yaml` and run
```
pip install -e .
```

## Usage

This will install two scripts:
```
view_data /path/to/folder
```
that can be used to visualise training data and predictions stored in our training data layout.

```
submit_slurm <SCRIPT_NAME> <SCRIPT_ARGS>
```
to run an arbitrary script on the cluster gpu queue.
