from setuptools import setup, find_packages

setup(
        name = 'grgrlib',
        # version = '0.0.0alpha',
        version = 'alpha',
        author='Gregor Boehl',
        author_email='admin@gregorboehl.com',
        description='Various functions & libraries for economic analysis',
        packages = find_packages(),
        install_requires=[
            'sympy',
            'scipy',
            'numpy',
            'pandas',
            'filterpy-dsge',
            'emcee',
            'pathos',
            'pydsge',
         ],
   )
