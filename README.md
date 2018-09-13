# grgrlib

Library of standard functions serving several processes. Still undocumented but straigtforward. 

Contains the functions and classes for solving, filtering and estimating DSGE models @ZLB.

The related code is in alpha state and provided for reasons of replicability and code sharing in the spirit of open science. It does not (and for now, can not) have a toolbox character. The code is operational, but (yet) not ready for public use and I can not provide any support. You are however very welcome to get in touch if you are interested working with the package.

The beta stage will involve considerable restructuring of packages, code, and the API.

The dependencies are listed in the setup.py file. Note that this package depends on the `pydsge` and `filterpy-dsge` packages that can both be found on my github page (they will thus not be installed automatically via `pip`).

## Installation with `pip`

Clone the repository and then from within the cloned folder run.
```
pip3 install .
```

