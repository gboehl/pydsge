#!/bin/python
# -*- coding: utf-8 -*-

import time
import pathos
import os.path
import tqdm
import numpy as np
# import cloudpickle as cpickle
import dill as cpickle


@property
def mask(self, verbose=False):

    if verbose:
        print('[mask:]'.ljust(15, ' ') + 'Shocks:', self.shocks)

    msk = self.data.copy()
    msk[:] = np.nan

    try:
        self.observables
    except AttributeError:
        self.get_sys(self.par, verbose=verbose)

    return msk.rename(columns=dict(zip(self.observables, self.shocks)))[:-1]


def parallellizer(sample, func_dump, verbose=True, ncores=None, **args):
    """Runs the `func_dump` in parallel. 

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

    func = cpickle.loads(func_dump)

    global runner
    func = runner

    def runner_loc(x):
        return func(x, **args)

    mapper = map
    if ncores is None or ncores > 1:
        pool = pathos.pools.ProcessPool(ncores)
        pool.clear()
        mapper = pool.imap

    wrap = tqdm.tqdm if verbose else lambda x: x

    res = wrap(mapper(runner_loc, sample), unit=' sample(s)',
               total=len(sample), dynamic_ncols=True)

    if ncores is None or ncores > 1:
        pool.close()
        pool.join()

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
        sample = self.get_par('prior', nsample=k, seed=seed, ncores=ncores, verbose=verbose)

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

    sample = get_sample(self, source=source, k=k, seed=seed,
                        ncores=ncores, verbose=verbose)

    global runner

    def runner(par):

        self.set_par(par, autocompile=False)
        self.get_sys(self.par, verbose=verbose > 1)
        self.preprocess(l_max=3, k_max=16, verbose=verbose > 1)

        self.filter.Q = self.QQ(self.par) @ self.QQ(self.par)

        FX = self.run_filter(verbose=False)

        mean, cov, eps, flag = self.extract(verbose=False, ngen=200, npop=5)

        if flag:
            print('[sampled_extract:]'.ljust(
                15, ' ') + 'Extract returned error.')

        return mean, cov, eps

    print('[sampled_extract:]'.ljust(15, ' ') + 'Starting extaction of shocks from %s...' %source)

    runner_dump = cpickle.dumps(runner)
    means, covs, eps = parallellizer(list(sample)[:k], runner_dump, ncores=ncores)

    print('[sampled_extract:]'.ljust(15, ' ') + 'Done extaction of shocks from %s...' %source)

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
            self, source=source, k=k, seed=seed, ncores=ncores, verbose=verbose)
    else:
        raise NotImplementedError('No other sampling methods implemented.')

    global runner

    def runner(arg, mask):

        par, eps, inits = arg

        self.set_par(par, autocompile=False)
        self.get_sys(self.par, verbose=verbose)
        self.preprocess(verbose=verbose)

        res = self.simulate(eps=eps, mask=mask, state=inits, verbose=verbose)

        return res

    runner_dump = cpickle.dumps(runner)
    res = parallellizer(list(
        zip(sample[:k], eps[:k], means[:k, 0])), runner_dump, mask=mask, ncores=ncores)

    return res


def sampled_irfs(self, shocklist, k=1, source=None, seed=None, ncores=None, verbose=False, **irfsargs):

    if source is None:
        source = 'posterior'
    sample = get_sample(self, source=source, k=k, seed=seed,
                        ncores=ncores, verbose=verbose)

    global runner

    def runner(par):

        self.set_par(par, autocompile=False)
        self.preprocess(verbose=verbose)

        res = self.irfs(shocklist, wannasee='full', verbose=verbose, **irfsargs)

        return res[0], res[2][1:]

    runner_dump = cpickle.dumps(runner)
    res = parallellizer(list(sample)[:k], runner_dump, ncores=ncores)

    return res
