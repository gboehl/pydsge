#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import pathos
import time
import tqdm
from .core import get_par


def mcmc(self, p0=None, nsteps=3000, nwalks=None, tune=None, moves=None, temp=False, seed=None, backend=True, suffix=None, linear=None, resume=False, append=False, update_freq=None, lprob_seed=None, biject=False, report=None, verbose=False, debug=False, **samplerargs):

    import pathos
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

    self.fdict['biject'] = biject

    from grgrlib.core import serializer

    lprob_global = serializer(self.lprob)

    if isinstance(temp, bool) and not temp:
        temp = 1

    def lprob(par): return lprob_global(
        par, linear, verbose, temp, lprob_seed or 'set')

    bnd = np.array(self.fdict['prior_bounds'])

    def bjfunc(x):
        if not biject:
            return x
        x = 1/(1 + np.exp(x))
        return (bnd[1] - bnd[0])*x + bnd[0]

    def rjfunc(x):
        if not biject:
            return x
        x = (x - bnd[0])/(bnd[1] - bnd[0])
        return np.log(1/x - 1)

    def lprob_scaled(x): return lprob(bjfunc(x))

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

        backend = emcee.backends.HDFBackend(backend)

        if not (resume or append):
            if not nwalks:
                raise TypeError(
                    "If neither `resume`, `append` or `p0` is given I need to know the number of walkers (`nwalks`).")
            try:
                backend.reset(nwalks, self.ndim)
            except KeyError as e:
                raise KeyError(
                    str(e) + '. Your `*.h5` file is likeli to be damaged...')
    else:
        backend = None

    if resume:
        nwalks = backend.get_chain().shape[1]

    if debug:
        sampler = emcee.EnsembleSampler(nwalks, self.ndim, lprob_scaled)
    else:
        sampler = emcee.EnsembleSampler(
            nwalks, self.ndim, lprob_scaled, moves=moves, pool=self.pool, backend=backend)

    if resume:
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

    p0 = rjfunc(p0) if biject else p0
    old_tau = np.inf
    cnt = 0

    for result in sampler.sample(p0, iterations=nsteps, **samplerargs):

        # cnt = sampler.iteration

        if not verbose:
            lls = list(result)[1]
            maf = np.mean(sampler.acceptance_fraction[-update_freq:])*100
            pbar.set_description('[ll/MAF:%s(%1.0e)/%1.0f%%]' %
                                 (str(np.max(lls))[:7], np.std(lls), maf))

        if cnt and update_freq and not cnt % update_freq:

            prnttup = '[mcmc:]'.ljust(
                15, ' ') + "Summary from last %s of %s iterations" % (update_freq, cnt)

            if temp < 1:
                prnttup += ' with temp of %s%%' % (np.round(temp, 6)*100)

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

            self.mcmc_summary(chain=bjfunc(sample), tune=update_freq,
                              calc_mdd=False, calc_ll_stats=True, out=lambda x: report(str(x)))

            report("Convergence stats: tau is in (%s,%s) (%s%s) and change is %s (%s0.01)." % (
                min_tau, max_tau, tau_sign, sampler.iteration/50, dev_tau.round(3), dev_sign))

        if cnt and update_freq and not (cnt+1) % update_freq:
            sample = sampler.get_chain()
            old_tau = emcee.autocorr.integrated_time(sample, tol=0)

        if not verbose:
            pbar.update(1)

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
    mode_x = bjfunc(chain[arg_max].flatten())

    self.fdict['mcmc_mode_x'] = mode_x
    self.fdict['mcmc_mode_f'] = mode_f

    if temp == 1:
        if 'mode_f' in self.fdict.keys() and mode_f < self.fdict['mode_f']:
            print('[mcmc:]'.ljust(15, ' ') + "New mode of %s is below old mode of %s. Rejecting..." %
                  (mode_f, self.fdict['mode_f']))
        else:
            self.fdict['mode_x'] = mode_x
            self.fdict['mode_f'] = mode_f

    return


def kdes(self, p0=None, nsteps=3000, nwalks=None, tune=None, seed=None, linear=None, resume=False, verbose=False, debug=False):

    import pathos
    import kombine
    from grgrlib.patches import kombine_run_mcmc

    kombine.Sampler.run_mcmc = kombine_run_mcmc

    if not hasattr(self, 'ndim'):
        # if it seems to be missing, lets do it.
        # but without guarantee...
        self.prep_estim(load_R=True)

    if seed is None:
        seed = self.fdict['seed']

    np.random.seed(seed)

    if tune is None:
        self.tune = None

    if linear is None:
        linear = self.filter.name == 'KalmanFilter'

    if nwalks is None:
        nwalks = 120

    if 'description' in self.fdict.keys():
        self.description = self.fdict['description']

    if not use_cloudpickle:
        # globals are *evil*
        global lprob_global
    else:
        import cloudpickle as cpickle
        lprob_dump = cpickle.dumps(self.lprob)
        lprob_global = cpickle.loads(lprob_dump)

    def lprob(par): return lprob_global(par, linear, verbose)

    if self.pool:
        self.pool.clear()

    if debug:
        sampler = kombine.Sampler(nwalks, self.ndim, lprob)
    else:
        sampler = kombine.Sampler(
            nwalks, self.ndim, lprob, pool=self.pool)

        if self.pool:
            self.pool.close()

    if p0 is not None:
        pass
    elif resume:
        # should work, but not tested
        p0 = self.fdict['kdes_chain'][-1]
    else:
        p0 = get_par(self, 'best', asdict=False, full=True,
                     nsample=nwalks, verbose=verbose)

    if not verbose:
        np.warnings.filterwarnings('ignore')

    if not verbose:
        pbar = tqdm.tqdm(total=nsteps, unit='sample(s)', dynamic_ncols=True)

    if nsteps < 500:
        nsteps_burnin = nsteps
        nsteps_mcmc = 0
    elif nsteps < 1000:
        nsteps_burnin = 500
        nsteps_mcmc = nsteps - nsteps_burnin
    else:
        nsteps_mcmc = 500
        nsteps_burnin = nsteps - nsteps_mcmc

    tune = max(500, nsteps_burnin)

    p, post, q = sampler.burnin(
        p0, max_steps=nsteps_burnin, pbar=pbar, verbose=verbose)

    if nsteps_mcmc:
        p, post, q = sampler.run_mcmc(nsteps_mcmc, pbar=pbar)

    acls = np.ceil(
        2/np.mean(sampler.acceptance[-tune:], axis=0) - 1).astype(int)
    samples = np.concatenate(
        [sampler.chain[-tune::acl, c].reshape(-1, 2) for c, acl in enumerate(acls)])

    # samples = sampler.get_samples()

    kdes_chain = sampler.chain
    kdes_sample = samples.reshape(1, -1, self.ndim)

    self.kdes_chain = kdes_chain
    self.kdes_sample = kdes_sample
    self.fdict['tune'] = tune
    self.fdict['kdes_chain'] = kdes_chain
    self.fdict['kdes_sample'] = kdes_sample

    pbar.close()

    if not verbose:
        np.warnings.filterwarnings('default')

    log_probs = sampler.get_log_prob()[self.tune:]
    chain = sampler.get_chain()[self.tune:]
    chain = chain.reshape(-1, chain.shape[-1])

    arg_max = log_probs.argmax()
    mode_f = log_probs.flat[arg_max]
    mode_x = chain[arg_max]

    self.fdict['kombine_mode_x'] = mode_x
    self.fdict['kombine_mode_f'] = mode_f

    if 'mode_f' in self.fdict.keys() and mode_f < self.fdict['mode_f']:
        print('[kombine:]'.ljust(15, ' ') + "New mode of %s is below old mode of %s. Rejecting..." %
              (mode_f, self.fdict['mode_f']))
    else:
        self.fdict['mode_x'] = mode_x
        self.fdict['mode_f'] = mode_x

    self.sampler = sampler

    return


def tmcmc(self, nsteps, nwalks, ntemps, target, update_freq=False, verbose=True, **mcmc_args):
    """Run Tempered Ensemble MCMC

    Parameters
    ----------
    ntemps : int
    target : float
    nsteps : float
    """

    from grgrlib.core import map2arr
    from .core import prior_sampler

    update_freq = update_freq if update_freq <= nsteps else False

    # sample pars from prior
    pars = prior_sampler(self, nwalks, test_lprob=False, verbose=verbose)

    x = get_par(self, 'prior_mean', asdict=False,
                full=False, verbose=verbose > 1)

    pbar = tqdm.tqdm(total=ntemps, unit='temp(s)', dynamic_ncols=True)
    sweat = False
    tmp = 0

    for i in range(ntemps):

        # update tmp
        ll = self.lprob(x)
        # treat first temperature increases extra carefully...
        lp = min(self.lprior(x), 0)
        # li = ll - lp

        # tmp = (target - lp)/li
        tmp = tmp*(ntemps-i-1)/(ntemps-i) + (target - lp)/(ntemps-i)/(ll - lp)
        aim = lp + (ll - lp)*tmp

        if tmp >= 1:
            # print only once
            if not sweat:
                pbar.write('[tmcmc:]'.ljust(
                    15, ' ') + "Increasing temperature to %s°. Too hot! I'm out..." % np.round(100*tmp, 3))
            sweat = True
            # skip for-loop to exit
            continue

        pbar.write('[tmcmc:]'.ljust(
            15, ' ') + "Increasing temperature to %2.5f°, aiming @ %4.3f." % (100*tmp, aim))
        pbar.set_description("[tmcmc: %2.3f°" % (100*tmp))

        self.mcmc(p0=pars, nsteps=nsteps, temp=tmp, update_freq=update_freq,
                  verbose=verbose > 1, append=i, report=pbar.write, **mcmc_args)

        self.temp = tmp
        self.mcmc_summary(tune=int(nsteps/10),
                          calc_mdd=False, calc_ll_stats=True)

        pbar.update()

        pars = self.get_chain()[-1]
        lprobs_adj = self.get_log_prob()[-1]
        x = pars[lprobs_adj.argmax()]

    pbar.close()

    return pars
