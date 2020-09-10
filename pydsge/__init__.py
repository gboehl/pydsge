#!/bin/python
# -*- coding: utf-8 -*-

import logging
import os
from .clsmethods import DSGE
from .gensys import gen_sys_from_dict as gen_sys
from .plots import sort_nhd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

logging.basicConfig(level=logging.INFO)

pth = os.path.dirname(__file__)

example_model = os.path.join(pth, 'examples', 'dfi.yaml')
example_data = os.path.join(pth, 'examples', 'tsdata.csv')
chain = os.path.join(pth, 'examples', 'dfi_doc0_sampler.h5')
meta_data = os.path.join(pth, 'examples', 'dfi_doc0_meta.npz')
res_dict = os.path.join(pth, 'examples', 'dfi_doc0_res.npz')

example = example_model, example_data
