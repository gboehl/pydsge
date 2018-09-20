# pydsge

Contains the functions and classes for solving, filtering and estimating DSGE models @ZLB. Undocumented and very rawwwww.

The code is in alpha state and provided for reasons of replicability and code sharing in the spirit of open science. It does not (and for now, can not) have a toolbox character. The code is operational, but (yet) not ready for public use and I can not provide any support. You are however very welcome to get in touch if you are interested working with the package.

The beta stage will involve considerable restructuring of packages, code, and the API.

The dependencies are listed in the setup.py file. Note that this package depends on the `econsieve` and `grgrlib` packages which both can be found on my github page (they will thus not be installed automatically via `pip`, at least for now).

## parser

The parser originally was a fork of Ed Herbst's fork from Pablo Winant's (excellent) package dolo. This version seemed slightly easier to adjust in order to obtain the matrices I need than in the up-to-date and more advanced version of dolo.

See https://github.com/EconForge/dolo and https://github.com/eph.

## Installation with `pip`

Clone the repository and then from within the cloned folder run.
```
pip3 install .
```

