#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import pathos
import time
import tqdm
from datetime import datetime
from .mpile import get_par


def mcmc(self, p0=None, nsteps=3000, nwalks=None, tune=None, moves=None, temp=False, seed=None, backend=True, suffix=None, linear=None, resume=False, append=False, update_freq=None, lprob_seed=None, report=None, maintenance_interval=10, verbose=False, debug=False, **samplerargs):

    import emcee

    if not hasattr(self, 'ndim'):
        # if it seems to be missing, lets do it.
        # but without guarantee...
        self.prep_estim(load_R=True)

    if seed is None:
        seed = self.fdict['seed']

    self.tune = tune
    if tune is None:
        self.tune = int(nsteps*1/5.)

    if update_freq is None:
        update_freq = int(nsteps/5.)

    if linear is None:
        linear = self.filter.name == 'KalmanFilter'

    if 'description' in self.fdict.keys():
        self.description = self.fdict['description']

    from grgrlib.multiprocessing import serializer

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    lprob_global = serializer(self.lprob)

    if isinstance(temp, bool) and not temp:
        temp = 1

    def lprob(par): return lprob_global(
        par, linear=linear, verbose=verbose, temp=temp, lprob_seed=lprob_seed or 'set')

    bnd = np.array(self.fdict['prior_bounds'])

    if self.pool:
        self.pool.clear()

    if p0 is None and not resume:
        if temp < 1:
            p0 = get_par(self, 'prior_mean', asdict=False,
                         full=False, nsample=nwalks, verbose=verbose)
        else:
            p0 = get_par(self, 'best', asdict=False, full=False,
                         nsample=nwalks, verbose=verbose)
    elif not resume:
        nwalks = p0.shape[0]

    if backend:

        if isinstance(backend, str):
            # backend_file will only be loaded later if explicitely defined before
            self.fdict['backend_file'] = backend
        try:
            backend = self.fdict['backend_file']
        except KeyError:
            # this is the default case
            suffix = str(suffix) if suffix else '_sampler.h5'
            backend = os.path.join(self.path, self.name+suffix)

            if os.path.exists(backend) and not (resume or append):
                print('[mcmc:]'.ljust(15, ' ') +
                      " HDF backend at %s already exists. Deleting..." % backend)
                os.remove(backend)

        backend = emcee.backends.HDFBackend(backend)

        if not (resume or append):
            if not nwalks:
                raise TypeError(
                    "If neither `resume`, `append` or `p0` is given I need to know the number of walkers (`nwalks`).")
            try:
                backend.reset(nwalks, self.ndim)
            except KeyError as e:
                raise KeyError(
                    str(e) + '. Your `*.h5` file is likely to be damaged...')
    else:
        backend = None

    if resume:
        nwalks = backend.get_chain().shape[1]

    if debug:
        sampler = emcee.EnsembleSampler(nwalks, self.ndim, lprob)
    else:
        sampler = emcee.EnsembleSampler(
            nwalks, self.ndim, lprob, moves=moves, pool=self.pool, backend=backend)

    if resume and not p0:
        p0 = sampler.get_last_sample()

    self.sampler = sampler
    self.temp = temp

    if not verbose:
        np.warnings.filterwarnings('ignore')

    if verbose > 2:
        report = report or print
    else:
        pbar = tqdm.tqdm(total=nsteps, unit='sample(s)', dynamic_ncols=True)
        report = report or pbar.write

    old_tau = np.inf
    cnt = 0

    for result in sampler.sample(p0, iterations=nsteps, **samplerargs):

        if not verbose:
            lls = list(result)[1]
            maf = np.mean(sampler.acceptance_fraction[-update_freq:])*100
            pbar.set_description('[ll/MAF:%s(%1.0e)/%1.0f%%]' %
                                 (str(np.max(lls))[:7], np.std(lls), maf))

        if cnt and update_freq and not cnt % update_freq:

            prnttup = '[mcmc:]'.ljust(
                15, ' ') + "Summary from last %s of %s iterations" % (update_freq, cnt)

            if temp < 1:
                prnttup += ' with temp of %s%%' % (np.round(temp*100, 6))

            if self.description is not None:
                prnttup += ' (%s)' % str(self.description)

            prnttup += ':'

            report(prnttup)

            sample = sampler.get_chain()

            tau = emcee.autocorr.integrated_time(sample, tol=0)
            min_tau = np.min(tau).round(2)
            max_tau = np.max(tau).round(2)
            dev_tau = np.max(np.abs(old_tau - tau)/tau)

            tau_sign = '>' if max_tau > sampler.iteration/50 else '<'
            dev_sign = '>' if dev_tau > .01 else '<'

            self.mcmc_summary(chain=sample, tune=update_freq,
                              calc_mdd=False, calc_ll_stats=True, out=lambda x: report(str(x)))

            report("Convergence stats: tau is in (%s,%s) (%s%s) and change is %s (%s0.01)." % (
                min_tau, max_tau, tau_sign, sampler.iteration/50, dev_tau.round(3), dev_sign))

        if cnt and update_freq and not (cnt+1) % update_freq:
            sample = sampler.get_chain()
            old_tau = emcee.autocorr.integrated_time(sample, tol=0)

        if not verbose:
            pbar.update(1)

        # avoid mem leakage
        if cnt and not cnt % maintenance_interval:
            self.pool.clear()

        cnt += 1

    pbar.close()
    if self.pool:
        self.pool.close()

    if not verbose:
        np.warnings.filterwarnings('default')

    log_probs = sampler.get_log_prob()[-self.tune:]
    chain = sampler.get_chain()[-self.tune:]
    chain = chain.reshape(-1, chain.shape[-1])

    arg_max = log_probs.argmax()
    mode_f = log_probs.flat[arg_max]
    mode_x = chain[arg_max].flatten()

    if temp == 1:

        self.fdict['mcmc_mode_x'] = mode_x
        self.fdict['mcmc_mode_f'] = mode_f

        if 'mode_f' in self.fdict.keys() and mode_f < self.fdict['mode_f']:
            print('[mcmc:]'.ljust(15, ' ') + " New mode of %s is below old mode of %s. Rejecting..." %
                  (mode_f, self.fdict['mode_f']))
        else:
            self.fdict['mode_x'] = mode_x
            self.fdict['mode_f'] = mode_f

    self.fdict['datetime'] = str(datetime.now())

    return


def tmcmc(self, nsteps, nwalks, ntemps, target, update_freq=False, test_lprob=False, verbose=True, debug=False, **mcmc_args):
    """Run Tempered Ensemble MCMC

    Parameters
    ----------
    ntemps : int
    target : float
    nsteps : float
    """

    from grgrlib.core import map2arr
    from .mpile import prior_sampler

    update_freq = update_freq if update_freq <= nsteps else False

    # sample pars from prior
    pars = prior_sampler(self, nwalks, test_lprob=test_lprob,
                         verbose=max(verbose, 2*debug))

    x = get_par(self, 'prior_mean', asdict=False,
                full=False, verbose=verbose > 1)

    pbar = tqdm.tqdm(total=ntemps, unit='temp(s)', dynamic_ncols=True)
    tmp = 0

    for i in range(ntemps):

        # update tmp
        ll = self.lprob(x)
        lp = self.lprior(x)

        tmp = tmp*(ntemps-i-1)/(ntemps-i) + (target - lp)/(ntemps-i)/(ll - lp)
        aim = lp + (ll - lp)*tmp

        if tmp >= 1:
            # print only once
            pbar.write('[tmcmc:]'.ljust(
                15, ' ') + "Increasing temperature to %s°. Too hot! I'm out..." % np.round(100*tmp, 3))
            pbar.update()
            self.temp = 1
            # skip for-loop to exit
            continue

        pbar.write('[tmcmc:]'.ljust(
            15, ' ') + "Increasing temperature to %2.5f°, aiming @ %4.3f." % (100*tmp, aim))
        pbar.set_description("[tmcmc: %2.3f°" % (100*tmp))

        self.mcmc(p0=pars, nsteps=nsteps, temp=tmp, update_freq=update_freq,
                  verbose=verbose > 1, append=i, report=pbar.write, debug=debug, **mcmc_args)

        self.temp = tmp
        self.mcmc_summary(tune=int(nsteps/10),
                          calc_mdd=False, calc_ll_stats=True)

        pbar.update()

        pars = self.get_chain()[-1]
        lprobs_adj = self.get_log_prob()[-1]
        x = pars[lprobs_adj.argmax()]

    pbar.close()
    self.fdict['datetime'] = str(datetime.now())

    return pars
