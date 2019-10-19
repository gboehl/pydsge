#!/bin/python
# -*- coding: utf-8 -*-

import time
import pathos
import os.path
import numpy as np
import tqdm


@property
def mask(self, verbose=False):

    if verbose:
        print('[mask:]'.ljust(15, ' ') + 'Shocks:', self.shocks)

    msk = self.data.copy()
    msk[:] = np.nan

    try:
        self.observables
    except AttributeError:
        self.get_sys()

    return msk.rename(columns=dict(zip(self.observables, self.shocks)))[:-1]


def parallellizer(sample, verbose=True, func=None, ncores=None, **args):
    """Runs global function `runner` in parallel. 

    Necessary for dill to avoid pickling of model objects. A dirty hack...

    Parameters
    ----------
    verbose : bool
        Whether to use tqdm to display process. Defaults to `True`.

    Returns
    -------
    tuple
        The result(s).
    """

    import pathos
    from grgrlib import map2arr

    global runner

    def runner_loc(x):
        return runner(x, **args)

    mapper = map
    if ncores is None or ncores > 1:
        pool = pathos.pools.ProcessPool(ncores)
        pool.clear()
        runner_loc = func
        mapper = pool.imap

    wrap = tqdm.tqdm if verbose else lambda x: x

    res = wrap(mapper(runner_loc, sample), unit=' sample(s)',
               total=len(sample), dynamic_ncols=True)

    return map2arr(res)


def get_sample(self, source=None, k=1, seed=None, ncores=None, verbose=False):
    """Creates (or loads) a parameter sample from `source`.

    If more samples are requested than already stored, new samples are taken.
    """

    if source is None and 'mcmc_mode_f' in self.fdict.keys():
        source = 'posterior'

    prefix = 'post_' if source is 'posterior' else 'prio_'
    try:
        sample_old = self.fdict[prefix+'sample']
        # don't use the same seed twice in one sample

        k -= sample_old.shape[0]
        if k < 1:
            return sample_old
        if seed is not None:
            seed += sample_old.shape[0]
    except:
        sample_old = None

    if source is 'posterior':

        import random

        random.seed(seed)
        sample = self.get_chain()[self.get_tune:]
        sample = sample.reshape(-1, sample.shape[-1])
        sample = random.choices(sample, k=k)

    else:
        sample = self.get_par('prior', nsample=k, seed=seed, ncores=ncores)

    if sample_old is not None:
        sample = np.concatenate((sample_old, sample), 0)

    self.fdict[prefix+'sample'] = sample

    return sample


def sampled_extract(self, source=None, k=1, seed=None, ncores=None, verbose=False):

    if source is None and 'mcmc_mode_f' in self.fdict.keys():
        source = 'posterior'

    prefix = 'post_' if source is 'posterior' else 'prio_'

    # adjust for source = 'random' to simulate with random noise given parameter

    try:
        means_old = self.fdict[prefix+'means']
        covs_old = self.fdict[prefix+'covs']
        eps_old = self.fdict[prefix+'eps']
        k -= eps_old.shape[0]
        if k < 1:
            return means_old, covs_old, eps_old
    except KeyError:
        eps_old = None

    sample = get_sample(self, source=source, k=k, seed=seed, ncores=ncores, verbose=verbose)

    global runner

    def runner(par):

        self.set_calib(par, autocompile=False)
        self.get_sys(par=par_active_lst, reduce_sys=True, verbose=verbose > 1)
        self.preprocess(l_max=3, k_max=16, verbose=verbose > 1)

        self.filter.Q = self.QQ(self.par) @ self.QQ(self.par)

        FX = self.run_filter(verbose=False)

        mean, cov, eps, flag = self.extract(verbose=False, ngen=200, npop=5)

        if flag:
            print('[sampled_extract:]'.ljust(
                15, ' ') + 'Extract returned error.')

        return mean, cov, eps

    means, covs, eps = parallellizer(sample, func=runner, ncores=ncores)

    if eps_old is not None:
        means = np.concatenate((means_old, means), 0)
        covs = np.concatenate((covs_old, covs), 0)
        eps = np.concatenate((eps_old, eps), 0)

    self.fdict[prefix+'means'] = means
    self.fdict[prefix+'covs'] = covs
    self.fdict[prefix+'eps'] = eps

    return means, covs, eps


def sampled_sim(self, k=1, source=None, mask=None, seed=None, ncores=None, verbose=False):

    if source is None:
        source = 'posterior'
    if source in ('prior', 'posterior'):
        sample = get_sample(self, source=source, k=k,
                            seed=seed, ncores=ncores, verbose=verbose)
        means, covs, eps = sampled_extract(
            self, source=source, k=k, seed=seed, verbose=verbose)
    else:
        raise NotImplementedError('No other sampling methods implemented.')

    global runner

    def runner(arg, mask):

        par, eps, inits = arg

        self.set_calib(par, autocompile=False)
        self.get_sys(verbose=verbose)
        self.preprocess(verbose=verbose)

        res = self.simulate(eps=eps, mask=mask, state=inits, verbose=verbose)

        return res

    res = parallellizer(list(zip(sample, eps, means[:, 0])), mask=mask, func=runner, ncores=ncores)

    return res


def sampled_irfs(self, shocklist, k=1, source=None, seed=None, ncores=None, verbose=False):

    if source is None:
        source = 'posterior'
    sample = get_sample(self, source=source, k=k, seed=seed, ncores=ncores, verbose=verbose)

    global runner

    def runner(par):

        self.set_calib(par, autocompile=False)
        self.preprocess(verbose=verbose)

        res = self.irfs(shocklist, wannasee='full', verbose=verbose)

        return res[0], res[2][1:]

    res = parallellizer(list(sample), func=runner, ncores=ncores)

    return res
