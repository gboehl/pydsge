#!/bin/python
# -*- coding: utf-8 -*-

# import numpy as np

# define additional functions used in the *.yaml.
# Of course, as this is a trivial function you could have defined it in the *.yaml directly
def calc_nu(nub):

    nu = nub / (1 - nub)

    return nu
