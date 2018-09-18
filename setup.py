from setuptools import setup, find_packages

setup(
        name = 'neatipy',
        version = 'alpha',
        author='Gregor Boehl',
        author_email='admin@gregorboehl.com',
        description='Solve, filter and estimate DSGE models with occasionaly binding constraints',
        packages = find_packages(),
        install_requires=[
            'sympy',
            'scipy',
            'numpy',
            'pandas',
            'econsieve',
            'emcee',
            'pathos',
            'pydsge',
         ],
   )
