from setuptools import setup, find_packages

setup(
    name="deep_cell.embl_tools",
    packages=find_packages(),
    version="0.0.1",
    author="Constantin Pape",
    url="https://github.com/constantinpape/deep-cell",
    license='MIT',
    entry_points={
        "console_scripts": [
            "view_data = utils_impl.view_data:main",
            "submit_slurm = utils_impl.submit_to_slurm:main"
        ]
    },
)
