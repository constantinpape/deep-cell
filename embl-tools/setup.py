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
            "view_data = visualisation.view_data:main",
            "submit_to_slurm = cluster.submit_to_slurm:main"
        ]
    },
)
