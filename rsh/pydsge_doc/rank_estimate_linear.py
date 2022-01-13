#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pydsge import DSGE
import emcee

yaml = 'rsh/pydsge_doc/rank.yaml'

mod = DSGE.read(yaml)  

mod.name = 'rank_test_linear'
mod.description = 'RANK, linear estimation'

mod.path = 'rsh/bs0/npz'

d0 = pd.read_csv('rsh/pydsge_doc/data.csv', sep=';', index_col='date', parse_dates=True).dropna() 

mod.load_data(d0, start='1983', end='2008Q4')

mod.prep_estim(linear=True, seed=0, verbose=True)

mod.filter.R = mod.create_obs_cov(1e-1)
ind = mod.observables.index('FFR')
mod.filter.R[ind,ind] /= 1e1 

fmax = None

moves = [(emcee.moves.DEMove(), 0.8), 
         (emcee.moves.DESnookerMove(), 0.2),]

p0 = mod.tmcmc(200, 200, 0, fmax, moves=moves, update_freq=100, lprob_seed='set')
mod.save()

mod.mcmc(p0, moves=moves, nsteps=2000, tune=500, update_freq=500, lprob_seed='set', append=True)
mod.save()

pars = mod.get_par('posterior', nsamples=250, full=True)
epsd0 = mod.extract(pars, nsamples=1)
mod.save_rdict(epsd0)

mod.mode_summary()
mod.mcmc_summary()
