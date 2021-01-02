#!/bin/python
# -*- coding: utf-8 -*-

import logging
import os
import numpy as np
from .gensys import DSGE, gen_sys_from_dict
from .plots import sort_nhd

np.set_printoptions(threshold=np.inf)
logging.basicConfig(level=logging.INFO)

pth = os.path.dirname(__file__)

example_model = os.path.join(pth, 'examples', 'dfi.yaml')
example_data = os.path.join(pth, 'examples', 'tsdata.csv')
chain = os.path.join(pth, 'examples', 'dfi_doc0_sampler.h5')
meta_data = os.path.join(pth, 'examples', 'dfi_doc0_meta.npz')
res_dict = os.path.join(pth, 'examples', 'dfi_doc0_res.npz')

example = example_model, example_data
