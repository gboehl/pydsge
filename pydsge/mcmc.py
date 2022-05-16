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

    if hasattr(self, "debug") and self.debug:

        self.pool = None

    elif hasattr(self, "pool"):

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

    sampler = ew.run_mcmc(lprob, nsteps, p0, moves=moves, priors=self.prior, backend=backend, resume=resume, pool=self.pool,
                          description=self.description, prior_transform=self.bptrans, temp=temp, verbose=verbose, **kwargs)

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
