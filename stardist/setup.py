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
            "train_stardist_2d = stardist_impl.train_stardist_2d:main",
            "predict_stardist_2d = stardist_impl.predict_stardist_2d:main",
            "train_stardist_3d = stardist_impl.train_stardist_3d:main",
            "predict_stardist_3d = stardist_impl.predict_stardist_3d:main",
        ]
    },
)
