"""This file contains estimation script.

#!/bin/python
# -*- coding: utf-8 -*-
"""

import numpy as np
import pandas as pd
from pydsge import DSGE
import emcee

yaml = "/home/gboehl/rsh/pydsge_doc/rank.yaml"
# TODO 1: use example model provided with package. Potentially adjust if necessary

mod = DSGE.read(yaml)

mod.name = "rank_test"
mod.description = "RANK, crisis sample"

mod.path = "/home/gboehl/rsh/bs0/npz"

d0 = pd.read_csv(
    "/home/gboehl/rsh/pydsge_doc/data.csv", sep=";", index_col="date", parse_dates=True
).dropna()
# TODO 2: use example data provided with package instead
# (contains only three time series and no confidential information)

# adjust elb
zlb = mod.get_par("elb_level")
rate = d0["FFR"]
d0["FFR"] = np.maximum(rate, zlb)

mod.load_data(d0, start="1998Q1")

# crucial command. Do some documentation
mod.prep_estim(N=350, seed=0, verbose=True)
# HINT: probably start playing around with the linear estimation first
# cause it is faster for obvious reasons. See other script provided.

mod.filter.R = mod.create_obs_cov(1e-1)
ind = mod.observables.index("FFR")
mod.filter.R[ind, ind] /= 1e1

fmax = None

moves = [
    (emcee.moves.DEMove(), 0.8),
    (emcee.moves.DESnookerMove(), 0.2),
]

p0 = mod.tmcmc(200, 200, 0, fmax, moves=moves, update_freq=100, lprob_seed="set")
mod.save()

mod.mcmc(
    p0,
    moves=moves,
    nsteps=3000,
    tune=500,
    update_freq=500,
    lprob_seed="set",
    append=True,
)
mod.save()

pars = mod.get_par("posterior", nsamples=250, full=True)
epsd0 = mod.extract(pars, nsamples=1)
mod.save_rdict(epsd0)

mod.mode_summary()
mod.mcmc_summary()
