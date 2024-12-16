from setuptools import find_packages, setup

setup(
    name="Transformers_for_coarse_graining",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        # Add other dependencies here
    ],
)