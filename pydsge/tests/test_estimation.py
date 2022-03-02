"""This file contains the test for the estimation of the model."""


def test_estimation():

    import numpy as np
    import pandas as pd
    from pydsge import DSGE, example

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

    sampler = mod.mcmc(p0, nsteps=10, tune=5, update_freq=5)
    # mod.save()

    mod.mcmc_summary()

    assert np.allclose(mod.sampler.get_chain()[0, 0], np.array(
        [0.79731672,  2.18496137,  1.58360637, -0.12263003,  0.09339783, 0.24715758,  0.83388983,  0.60519207,  0.26469191,  0.0791741, 0.89252174]))
