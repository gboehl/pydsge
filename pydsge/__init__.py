#!/bin/python
# -*- coding: utf-8 -*-

import os
from .clsmethods import DSGE, get_data

pth = os.path.dirname(__file__)
yamlpth = os.path.join(pth,'..','tests','nk.yaml')
nk_model = DSGE.read(yamlpth)
