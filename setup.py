from setuptools import setup, find_packages

setup(
        name = 'pydsge',
        version = 'alpha',
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
            'pyyaml',
         ],
   )
