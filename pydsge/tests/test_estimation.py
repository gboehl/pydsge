"""This file contains the test for the estimation of the model."""

import os
import numpy as np
import pandas as pd
from pydsge import DSGE, example


def test_estimation(create=False):

    yaml, data = example
    mod = DSGE.read(yaml)

    d0 = pd.read_csv(data, index_col='date', parse_dates=True)

    # adjust elb
    zlb = mod.get_par('elb_level')
    rate = d0['FFR']
    d0['FFR'] = np.maximum(rate, zlb)

    mod.load_data(d0, start='1998Q1')

    mod.prep_estim(N=150, seed=0, verbose=True)

    mod.filter.R = mod.create_obs_cov(1e-1)
    ind = mod.observables.index('FFR')
    mod.filter.R[ind, ind] /= 1e1

    # sample pars from prior
    p0 = mod.prior_sampler(22, check_likelihood=False, verbose=True)

    sampler = mod.mcmc(p0, nsteps=20, tune=5, update_freq=0)
    # mod.save()

    mod.mcmc_summary()
    this_sample = mod.sampler.get_chain()[0, 0]

    pth = os.path.dirname(__file__)
    save_path = os.path.join(pth, "resources", "estimation.npz")

    if create:
        with open(save_path, "wb") as f:
            np.save(f, this_sample)

    else:
        with open(save_path, "rb") as f:

            test_sample = np.load(f, allow_pickle=False)
            np.testing.assert_allclose(this_sample, test_sample)
