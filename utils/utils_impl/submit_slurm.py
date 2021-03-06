#! /usr/bin/python3

import os
import sys
import inspect
import subprocess
from datetime import datetime

# two days in minutes
TWO_DAYS = 2 * 24 * 60


def write_slurm_template(script, out_path, env_name,
                         n_threads, gpu_type, n_gpus,
                         mem_limit, time_limit, qos,
                         mail_address):
    slurm_template = ("#!/bin/bash\n"
                      "#SBATCH -N 1\n"
                      "#SBATCH -c %s\n"
                      "#SBATCH --mem %s\n"
                      "#SBATCH -t %i\n"
                      "#SBATCH --qos=%s\n"
                      "#SBATCH -p gpu\n"
                      "#SBATCH -C gpu=%s\n"
                      "#SBATCH --gres=gpu:%i\n") % (n_threads,
                                                    mem_limit, time_limit,
                                                    qos, gpu_type, n_gpus)
    if mail_address is not None:
        slurm_template += ("#SBATCH --mail-type=FAIL,BEGIN,END\n"
                           "#SBATCH --mail-user=%s") % mail_address
    slurm_template += ("\n"
                       "module purge \n"
                       "module load GCC \n"
                       "source activate %s\n"
                       "\n"
                       "python %s $@ \n") % (env_name, script)
    with open(out_path, 'w') as f:
        f.write(slurm_template)


def submit_slurm(script, input_, n_threads=7, n_gpus=1,
                 gpu_type='2080Ti|1080Ti', mem_limit='64G',
                 time_limit=TWO_DAYS, qos='normal',
                 env_name=None, mail_address=None):
    """ Submit python script that needs gpus with given inputs on a slurm node.
    """

    tmp_folder = os.path.expanduser('~/.deep-cell/slurm')
    os.makedirs(tmp_folder, exist_ok=True)

    print("Submitting training script %s to cluster" % script)
    print("with arguments %s" % " ".join(input_))

    script_name = os.path.split(script)[1]
    dt = datetime.now().strftime('%Y_%m_%d_%M')
    tmp_name = os.path.splitext(script_name)[0] + dt
    batch_script = os.path.join(tmp_folder, '%s.sh' % tmp_name)
    log = os.path.join(tmp_folder, '%s.log' % tmp_name)
    err = os.path.join(tmp_folder, '%s.err' % tmp_name)

    if env_name is None:
        env_name = os.environ.get('CONDA_DEFAULT_ENV', None)
        if env_name is None:
            raise RuntimeError("Could not find conda")

    print("Batch script saved at", batch_script)
    print("Log will be written to %s, error log to %s" % (log, err))
    write_slurm_template(script, batch_script, env_name,
                         int(n_threads), gpu_type, int(n_gpus),
                         mem_limit, int(time_limit), qos, mail_address)

    cmd = ['sbatch', '-o', log, '-e', err, '-J', script_name, batch_script]
    cmd.extend(input_)
    subprocess.run(cmd)


def scrape_kwargs(input_):
    params = inspect.signature(submit_slurm).parameters
    kwarg_names = [name for name in params
                   if params[name].default != inspect._empty]
    kwarg_positions = [i for i, inp in enumerate(input_)
                       if inp in kwarg_names]

    kwargs = {input_[i]: input_[i + 1] for i in kwarg_positions}

    kwarg_positions += [i + 1 for i in kwarg_positions]
    input_ = [inp for i, inp in enumerate(input_) if i not in kwarg_positions]

    return input_, kwargs


def main():
    script = os.path.realpath(os.path.abspath(sys.argv[1]))
    input_ = sys.argv[2:]

    # scrape the additional arguments (n_threads, mem_limit, etc. from the input)
    input_, kwargs = scrape_kwargs(input_)
    submit_slurm(script, input_, **kwargs)


if __name__ == '__main__':
    main()
