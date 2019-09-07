from setuptools import setup, find_packages

setup(
        name = 'pydsge',
        version = '0.0.1',
        author='Gregor Boehl',
        author_email='admin@gregorboehl.com',
        description='Solve, filter and estimate DSGE models with occasionaly binding constraints',
        packages = find_packages(),
        install_requires=[
            'econsieve',
            'emcee',
            'numba',
            'numpy',
            'pandas',
            'pathos',
            'scipy',
            'sympy',
            'tqdm',
            'pyaml',
            'pygmo',
         ],
   )
