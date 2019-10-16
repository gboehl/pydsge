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


def parallellizer(sample, ncores=None, verbose=True, **args):

    import pathos
    import tqdm
    from grgrlib import map2arr

    global runner

    def runner_loc(x):
        return runner(x, **args)

    pool = pathos.pools.ProcessPool()
    pool.clear()

    wrap = tqdm.tqdm if verbose else lambda x: x

    res = wrap(pool.imap(runner_loc, sample), unit=' sample(s)',
               total=len(sample), dynamic_ncols=True)

    return map2arr(res)


def get_sample(self, source=None, k=1, seed=None, verbose=False):

    if source is None and 'mcmc_mode_f' in self.fdict.keys():
        source = 'posterior'

    if source is 'posterior':

        import random

        random.seed(seed)
        sample = self.get_chain()[self.get_tune:]
        sample = sample.reshape(-1, sample.shape[-1])
        sample = random.choices(sample, k=ke)

        return sample

    return self.get_par('priors', nsample=ke, seed=seed)


def sampled_extract(self, source=None, k=1, seed=None, verbose=False):

    if source is None and 'mcmc_mode_f' in self.fdict.keys():
        source = 'posterior'

    prefix = 'post_' if source is 'posterior' else 'prio_'

    try:
        means_old = self.fdict[prefix+'means']
        covs_old = self.fdict[prefix+'covs']
        eps_old = self.fdict[prefix+'eps']
        sam_old = self.fdict[prefix+'sample']
        ke = k-eps_old.shape[0]
    except KeyError:
        eps_old = None
        ke = k

    if ke < 1:
        return

    sample = get_sample(self, source=source, k=ke, seed=seed, verbose=verbose)

    global runner

    def runner(par):

        par_fix = self.par_fix
        par_fix[self.prior_arg] = par
        par_active_lst = list(par_fix)

        self.get_sys(par=par_active_lst, reduce_sys=True, verbose=verbose > 1)

        self.preprocess(l_max=3, k_max=16, verbose=verbose > 1)
        self.filter.Q = self.QQ(self.par) @ self.QQ(self.par)

        FX = self.run_filter(verbose=False)

        mean, cov, eps, flag = self.extract(verbose=False, ngen=200, npop=5)

        if flag:
            print('[sampled_extract:]'.ljust(
                15, ' ') + 'Extract returned error.')

        return mean, cov, eps

    means, covs, eps = parallellizer(sample)

    if eps_old is not None:
        means = np.vstack((means_old, means))
        covs = np.vstack((covs_old, covs))
        eps = np.vstack((eps_old, eps))
        sample = np.vstack((sam_old, sample))

    self.fdict[prefix+'means'] = means
    self.fdict[prefix+'covs'] = covs
    self.fdict[prefix+'eps'] = eps
    self.fdict[prefix+'sample'] = sample

    return


def sampled_sim(self, source=None, mask=None, ncores=None, verbose=False):

    if source is not None:
        prefix = 'post_' if source is 'posterior' else 'prio_'

        eps = self.fdict[prefix+'eps']
        sample = self.fdict[prefix+'sample']

    else:
        try:
            prefix = 'post_'
            eps = self.fdict[prefix+'eps']
            sample = self.fdict[prefix+'sample']
        except:
            prefix = 'prio_'
            eps = self.fdict[prefix+'eps']
            sample = self.fdict[prefix+'sample']

    global runner

    def runner(arg, mask):

        par, eps = arg

        self.set_parval(par)
        self.get_sys(verbose=verbose)
        self.preprocess(verbose=verbose)

        res = self.simulate(eps, mask, verbose=verbose)

        return res

    res = parallellizer(list(zip(sample, eps)), ncores=ncores, mask=mask)

    return res


def sampled_irfs(self, shocklist, nbatch=1, wannasee=None, source=None, ncores=None, verbose=False):

    # this should load existing samples as well
    sample = get_sample(self, source=source, k=nbatch,
                        seed=seed, verbose=verbose)

    global runner

    def runner(par, shocklist, wannasee):

        self.set_parval(par)
        self.get_sys(verbose=verbose)
        self.preprocess(verbose=verbose)

        res = self.irfs(shocklist, wannasee, verbose=verbose)

        return res

    res = parallellizer(list(sample), ncores=ncores,
                        shocklist=shocklist, wannasee=wannasee)

    return res
