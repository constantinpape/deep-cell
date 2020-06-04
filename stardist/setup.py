from setuptools import setup, find_packages

setup(
    name="deep_cell.stardist",
    packages=find_packages(),
    version="0.0.1",
    author="Constantin Pape",
    url="https://github.com/constantinpape/deep-cell",
    license='MIT',
    entry_points={
        "console_scripts": [
            "train_stardist_2d = training.train_stardist_2d:main",
            "predict_stardist = prediction.predict_stardist:main"
        ]
    },
)
