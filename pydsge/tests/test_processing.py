"""This file contains the test for the processing estimation results."""


def test_processing():

    import matplotlib.pyplot as plt
    # import the base class:
    from pydsge import *
    # import all the useful stuff from grgrlib:
    from grgrlib import *

    mod = DSGE.load(meta_data)
    info = mod.info()
    summary = mod.mcmc_summary()

    pars = mod.get_par('posterior', nsamples=100, full=True)

    ir0 = mod.irfs(('e_r', 1, 0), pars)

    # # plot them:
    v = ['y', 'Pi', 'r', 'x']
    fig, ax, _ = pplot(ir0[0][..., mod.vix(v)], labels=v)

    pars_sig1 = [mod.set_par('sigma', 1, p) for p in pars]

    # load filter:
    mod.load_estim()
    # extract shocks:
    epsd = mod.extract(pars, nsamples=1, bound_sigma=4)
