"""This file defines the installation set-up."""
from setuptools import setup, find_packages
from os import path

# get version from dedicated version file
version = {}
with open("pydsge/__version__.py") as fp:
    exec(fp.read(), version)

# read the contents of the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://pydsge.readthedocs.io/en/latest/index.html",
    name="pydsge",
    version=version['__version__'],
    author="Gregor Boehl",
    author_email="admin@gregorboehl.com",
    license="MIT",
    description=(
        "Solve, filter and estimate DSGE models with occasionaly binding constraints"
    ),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    package_data={"pydsge": ["examples/*"]},
    install_requires=[
        "numba",
        "emcee",
        "numpy",
        "pandas",
        "pathos",
        "dill",
        "tqdm",
        "cloudpickle",
        "h5py",
        "pyyaml",
        "scipy",
        "sympy>=1.12",
        "emcwrap>=0.2.5",
        "grgrlib>=0.1.15",
        "econsieve>=0.0.9",
        "threadpoolctl>=3.1.0",
    ],
    include_package_data=True,
)
