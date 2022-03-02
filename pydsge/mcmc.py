#!/bin/python
# -*- coding: utf-8 -*-

import os
import time
import tqdm
import emcee
import numpy as np
import pandas as pd
import emcwrap as ew
from datetime import datetime
from grgrlib.multiprocessing import serializer
from .mpile import get_par


def mcmc(
    self,
    p0=None,
    nsteps=3000,
    nwalks=None,
    tune=None,
    moves=None,
    temp=False,
    seed=None,
    backend=True,
    suffix=None,
    resume=False,
    append=False,
    report=None,
    maintenance_interval=10,
    verbose=False,
    **kwargs
):
    """Run the emcee ensemble MCMC sampler.

    Parameters:
    ----------

    p0 : ndarray of initial states of the walkers in the parameterspace
    moves : emcee.moves object
    """

    if not hasattr(self, "ndim"):
        # if it seems to be missing, lets do it.
        # but without guarantee...
        self.prep_estim(load_R=True)

    if seed is None:
        seed = self.fdict["seed"]

    self.tune = tune
    if tune is None:
        self.tune = int(nsteps * 1 / 5.0)

    if "description" in self.fdict.keys():
        self.description = self.fdict["description"]

    if hasattr(self, "pool"):
        from .estimation import create_pool

        create_pool(self)

    lprob_global = serializer(self.lprob)

    if isinstance(temp, bool) and not temp:
        temp = 1

    linear = self.filter.name == "KalmanFilter"

    def lprob(par):
        return lprob_global(
            par,
            linear=linear,
            verbose=verbose,
            temp=temp
        )

    if self.pool:
        self.pool.clear()

    if p0 is None and not resume:
        if temp < 1:
            p0 = get_par(
                self,
                "prior_mean",
                asdict=False,
                full=False,
                nsample=nwalks,
                verbose=verbose,
            )
        else:
            p0 = get_par(
                self, "best", asdict=False, full=False, nsample=nwalks, verbose=verbose
            )
    elif not resume:
        nwalks = p0.shape[0]

    if backend:

        if isinstance(backend, str):
            # backend_file will only be loaded later if explicitely defined before
            self.fdict["backend_file"] = backend
        try:
            backend = self.fdict["backend_file"]
        except KeyError:
            # this is the default case
            suffix = str(suffix) if suffix else "_sampler.h5"
            backend = os.path.join(self.path, self.name + suffix)

            if os.path.exists(backend) and not (resume or append):
                print(
                    "[mcmc:]".ljust(15, " ")
                    + " HDF backend at %s already exists. Deleting..." % backend
                )
                os.remove(backend)

        backend = emcee.backends.HDFBackend(backend)

        if not (resume or append):
            if not nwalks:
                raise TypeError(
                    "If neither `resume`, `append` or `p0` is given I need to know the number of walkers (`nwalks`)."
                )
            try:
                backend.reset(nwalks, self.ndim)
            except KeyError as e:
                raise KeyError(
                    str(e) + ". Your `*.h5` file is likely to be damaged...")
    else:
        backend = None

    sampler = ew.run_mcmc(lprob, p0, nsteps, priors=self.prior, backend=backend, resume=resume,
                          pool=self.pool, description=self.description, temp=temp, verbose=verbose, **kwargs)

    self.temp = temp
    self.sampler = sampler

    if temp == 1:

        log_probs = sampler.get_log_prob()[-tune:]
        chain = sampler.get_chain()[-tune:]
        chain = chain.reshape(-1, chain.shape[-1])

        arg_max = log_probs.argmax()
        mode_f = log_probs.flat[arg_max]
        mode_x = chain[arg_max].flatten()

        self.fdict["mode_x"] = mode_x
        self.fdict["mode_f"] = mode_f

    self.fdict["datetime"] = str(datetime.now())

    return

# deprepciated and to be removed


def tmcmc(
    self,
    nwalks,
    nsteps=200,
    ntemps=0,
    target=None,
    update_freq=False,
    check_likelihood=False,
    verbose=True,
    debug=False,
    **mcmc_args
):
    """Run Tempered Ensemble MCMC

    Parameters
    ----------
    ntemps : int
    target : float
    nsteps : float
    """

    from grgrlib import map2arr
    from .mpile import prior_sampler

    update_freq = update_freq if update_freq <= nsteps else False

    # sample pars from prior
    pars = prior_sampler(
        self, nwalks, check_likelihood=check_likelihood, verbose=max(
            verbose, 2 * debug)
    )

    x = get_par(self, "prior_mean", asdict=False,
                full=False, verbose=verbose > 1)

    pbar = tqdm.tqdm(total=ntemps, unit="temp(s)", dynamic_ncols=True)
    tmp = 0

    for i in range(ntemps):

        # update tmp
        ll = self.lprob(x)
        lp = self.lprior(x)

        tmp = tmp * (ntemps - i - 1) / (ntemps - i) + (target - lp) / (ntemps - i) / (
            ll - lp
        )
        aim = lp + (ll - lp) * tmp

        if tmp >= 1:
            # print only once
            pbar.write(
                "[tmcmc:]".ljust(15, " ")
                + "Increasing temperature to %s°. Too hot! I'm out..."
                % np.round(100 * tmp, 3)
            )
            pbar.update()
            self.temp = 1
            # skip for-loop to exit
            continue

        pbar.write(
            "[tmcmc:]".ljust(15, " ")
            + "Increasing temperature to %2.5f°, aiming @ %4.3f." % (100 * tmp, aim)
        )
        pbar.set_description("[tmcmc: %2.3f°" % (100 * tmp))

        self.mcmc(
            p0=pars,
            nsteps=nsteps,
            temp=tmp,
            update_freq=update_freq,
            verbose=verbose > 1,
            append=i,
            report=pbar.write,
            debug=debug,
            **mcmc_args
        )

        self.temp = tmp
        self.mcmc_summary(tune=int(nsteps / 10),
                          calc_mdd=False, calc_ll_stats=True)

        pbar.update()

        pars = self.get_chain()[-1]
        lprobs_adj = self.get_log_prob()[-1]
        x = pars[lprobs_adj.argmax()]

    pbar.close()
    self.fdict["datetime"] = str(datetime.now())

    return pars
