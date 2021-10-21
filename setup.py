from setuptools import setup, find_packages
from os import path

# read the contents of the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
        long_description=long_description,
        long_description_content_type="text/x-rst",
        url="https://pydsge.readthedocs.io/en/latest/index.html",
        name = 'pydsge',
        version = '0.0.1',
        author='Gregor Boehl',
        author_email='admin@gregorboehl.com',
        license='MIT',
        description='Solve, filter and estimate DSGE models with occasionaly binding constraints',
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
        ],
        packages = find_packages(),
        package_data={'pydsge': ["examples/*"]},
        install_requires=[
            'emcee',
            'numba',
            'numpy',
            'pandas',
            'pathos',
            'scipy',
            'sympy',
            'tqdm',
            'pyaml',
            'grgrlib',
            'cloudpickle',
         ],
        include_package_data=True,
   )
