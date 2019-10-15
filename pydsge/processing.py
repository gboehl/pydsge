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

    return msk.rename(columns=dict(zip(self.observables, self.shocks)))[:-1]


def parallellizer(sample, ncores=None, verbose=True, *args):

    import pathos
    import tqdm
    from grgrlib import map2arr

    global runner

    def runner_loc(x):
        return runner(x, *args)

    pool = pathos.pools.ProcessPool()
    pool.clear()

    wrap = tqdm.tqdm if verbose else lambda x: x

    res = wrap(pool.imap(runner_loc, sample), unit=' sample(s)', total=len(sample), dynamic_ncols=True)

    return map2arr(res)


def sampled_extract(self, source=None, k=1, seed=None, verbose=False):

    if source is None and 'mcmc_mode_f' in self.fdict.keys():
        source = 'posterior'

    if source is 'posterior':

        import random 

        random.seed(seed)
        sample = self.get_chain()[self.get_tune:]
        sample = sample.reshape(-1, sample.shape[-1])
        sample = random.choices(sample, k=k)

    else:
        sample = self.get_par('priors', nsample=k, seed=seed)

    global runner
    
    def runner(par):

        par_fix = self.par_fix
        par_fix[self.prior_arg] = par
        par_active_lst = list(par_fix)

        self.get_sys(par=par_active_lst, reduce_sys=True, verbose=verbose > 1)

        self.preprocess(l_max=3, k_max=16, verbose=verbose > 1)
        self.filter.Q = self.QQ(self.par) @ self.QQ(self.par)

        FX = self.run_filter(verbose=False)

        SX, scov, eps, flag = self.extract(verbose=False, ngen=200, npop=5)
        
        if flag:
            print('[sampled_extract:]'.ljust(15, ' ') + 'Extract returned error.')

        return SX, scov, eps

    res = parallellizer(sample)

    self.fdict['eps'] = eps

    return res


"""
def sampled_sim(self, epstracted=None, mask=None, reduce_sys=None, forecast=False, linear=False, nr_samples=None, ncores=None, show_warnings=False, verbose=False):
    # rewrite!

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    if epstracted is None:
        epstracted = self.epstracted[:4]

    if reduce_sys is None:
        reduce_sys = self.is_reduced

    XX, COV, EPS, PAR = epstracted

    # XX had to be saved as an object array. pathos can't deal with that
    if XX.ndim > 2:
        X0 = [x.astype(float) for x in XX[:, 0, :]]
    else:
        X0 = [x.astype(float) for x in XX]

    if nr_samples is None:
        nr_samples = EPS.shape[0]

    if forecast:
        E0 = np.zeros((EPS.shape[0], forecast, EPS.shape[2]))
        EPS = np.hstack([EPS, E0])
        if mask is not None:
            m0 = np.zeros((forecast, EPS.shape[2]))
            mask = np.vstack([mask, m0])

    def runner(nr, mask):

        par = list(PAR[nr])
        eps = EPS[nr]
        x0 = X0[nr]

        self.get_sys(par, reduce_sys=reduce_sys, verbose=verbose)
        self.preprocess(verbose=verbose)

        if mask is not None:
            eps = np.where(np.isnan(mask), eps, mask*eps)

            ss = np.where(self.SIG)[0]
            mask1 = np.where(np.isnan(mask[0]), 1, mask[0])
            x0[ss] = self.SIG[ss] @ np.diag(mask1) @ self.SIG.T @ x0

        SX, SK, flag = self.simulate(
            eps, initial_state=x0, linear=linear, show_warnings=show_warnings, verbose=verbose, return_flag=True)

        SZ = (self.hx[0] @ SX.T).T + self.hx[1]

        return SZ, SX, SK, self.vv

    global runner_glob
    runner_glob = runner

    res = runner_pooled(nr_samples, ncores, mask, True)

    dim_x = 1e50
    for p in res:
        if len(p[3]) < dim_x:
            minr = p[3]
            dim_x = len(minr)

    SZS = []
    SXS = []
    SKS = []

    for n, p in enumerate(res):

        SZS.append(p[0])
        SKS.append(p[2])

        if len(p[3]) > dim_x:
            idx = [list(p[3]).index(v) for v in minr]
            SXS.append(p[1][:, idx])
        else:
            SXS.append(p[1])

    return np.array(SZS), np.array(SXS), np.array(SKS)


def sampled_irfs(self, be_res, shocklist, wannasee, reduce_sys=None, nr_samples=1000, ncores=None, show_warnings=False):
    # rewrite!

    import pathos

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    if reduce_sys is None:
        reduce_sys = self.is_reduced

    # dry run
    par = be_res.means()
    # define parameters
    self.get_sys(par, reduce_sys=reduce_sys, verbose=False)
    # preprocess matrices for speedup
    self.preprocess(verbose=False)

    Xlabels = self.irfs(shocklist, wannasee)[1]

    def runner(nr, useless_arg):

        par = self.posterior_sample(be_res=be_res, seed=nr)

        # define parameters
        self.get_sys(par, reduce_sys=reduce_sys, verbose=False)
        # preprocess matrices for speedup
        self.preprocess(verbose=False)

        xs, _, (ys, ks, ls) = self.irfs(
            shocklist, wannasee, show_warnings=show_warnings)

        return xs, ys, ks, ls

    global runner_glob
    runner_glob = runner

    res = runner_pooled(nr_samples, ncores, None, True)

    X, Y, K, L = [], [], [], []

    for p in res:
        X.append(p[0])
        Y.append(p[1])
        K.append(p[2])
        L.append(p[3])

    return np.array(X), Xlabels, (np.array(Y), np.array(K), np.array(L))
"""
