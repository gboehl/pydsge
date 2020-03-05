#!/bin/python
# -*- coding: utf-8 -*-

import os
from .clsmethods import DSGE
from .plots import sort_nhd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import logging
logging.basicConfig(level=logging.INFO)

pth = os.path.dirname(__file__)

example_model = os.path.join(pth, 'examples', 'dfi.yaml')
example_data = os.path.join(pth, 'examples', 'tsdata.csv')

example = example_model, example_data
