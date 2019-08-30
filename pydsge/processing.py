#!/bin/python
# -*- coding: utf-8 -*-
import warnings
import time
import emcee
import pathos
import os.path
import numpy as np
from tqdm import tqdm
from .parser import DSGE as dsge



def mask(self, verbose=False):
    if verbose:
        print('[mask:]'.ljust(15, ' ') + 'Shocks:', self.shocks)
    return np.full((self.Z.shape[0]-1, self.Z.shape[1]), np.nan)

def runner_pooled(nr_samples, ncores, mask, use_pbar):

    import pathos

    global runner_glob

    def runner_loc(x):
        return runner_glob(x, mask)

    pool = pathos.pools.ProcessPool(ncores)

    if use_pbar:
        res = list(tqdm(pool.imap(runner_loc, range(nr_samples)),
                        unit=' sample(s)', total=nr_samples, dynamic_ncols=True))
    else:
        res = list(pool.imap(runner_loc, range(nr_samples)))

    pool.close()
    pool.join()
    pool.clear()

    return res


def posterior_sample(self, be_res=None, seed=0, verbose=False):

    import random

    if be_res is None:
        chain = self.sampler.get_chain()
        tune = self.sampler.tune
        par_fix = self.par_fix
        prior_arg = self.prior_arg,
        prior_names = self.prior_names

    else:
        chain = be_res.chain
        tune = be_res.tune
        par_fix = be_res.par_fix
        prior_arg = be_res.prior_arg
        prior_names = be_res.prior_names

    all_pars = chain[tune:].reshape(-1, chain.shape[-1])

    random.seed(seed)
    randpar = par_fix
    psample = random.choice(all_pars)
    randpar[prior_arg] = psample

    if verbose:
        pstr = ''
        for pv, pn in zip(psample, prior_names):
            if pstr:
                pstr += ', '
            pstr += pn + ': ' + str(pv.round(3))

        print('[epstract:]'.ljust(15, ' ') +
              'Parameters drawn from posterior:')
        print(''.ljust(15, ' ') + pstr)

    return list(randpar)


def epstract(self, be_res=None, N=None, nr_samples=100, save=None, ncores=None, method=None, itype=(0, 1), penalty=10, max_attempts=3, presmoothing=None, min_options=None, reduce_sys=False, force=False, verbose=False):

    XX = []
    COV = []
    EPS = []
    PAR = []

    if N is None:
        N = 500

    if reduce_sys is not self.is_reduced:
        self.get_sys(reduce_sys=reduce_sys)

    if not force and save is not None and os.path.isfile(save):

        files = np.load(save, allow_pickle=True)
        OBS_COV = files['OBS_COV']
        smethod = files['method']
        mess1 = ''
        mess2 = ''

        if 'is_reduced' in files:
            if reduce_sys is not files['is_reduced']:
                mess1 = ", epstract overwrites 'reduce_sys'"

            reduce_sys = files['is_reduced']
        if 'presmoothing' in files:
            presmoothing = files['presmoothing'].item()

        EPS = files['EPS']
        COV = files['COV']
        XX = files['XX']
        PAR = files['PAR']

        if EPS.shape[0] >= nr_samples:
            if presmoothing is not None:
                mess2 = 'presmoothing: %s,' % presmoothing

            print('[epstract:]'.ljust(15, ' ') +
                  'Epstract already exists (%sreduced: %s%s)' % (mess2, reduce_sys, mess1))

            self.epstracted = XX, COV, EPS, PAR, self.obs_cov, method

            return XX, COV, EPS, PAR

        else:
            print('[epstract:]'.ljust(15, ' ') +
                  'Appending to existing epstract...')

            XX = list(XX)
            COV = list(COV)
            EPS = list(EPS)
            PAR = list(PAR)

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    yet = 0
    if len(EPS):
        yet = len(EPS)
    nr_samples = nr_samples - yet

    def runner(nr, mask):

        flag = True

        for att in range(max_attempts):

            par = self.posterior_sample(
                be_res=be_res, seed=yet + nr + att*nr_samples)

            # define parameters
            self.get_sys(par, reduce_sys=reduce_sys, verbose=False)
            # preprocess matrices for speedup
            self.preprocess(verbose=False)

            self.create_filter(N=N)
            SX, scov = self.run_filter()
            IX, icov, eps, flag = self.extract(method=method, penalty=penalty, return_flag=True, itype=itype,
                                               presmoothing=presmoothing, min_options=min_options, show_warnings=False, verbose=verbose)

            if not flag:
                return IX, icov, eps, par

        raise ValueError(
            'epstract: Could not extract shocks after %s attemps.' % max_attempts)

    global runner_glob
    runner_glob = runner

    res = runner_pooled(nr_samples, ncores, None, not verbose)

    no_obs, dim_z = self.Z.shape
    dim_e = len(self.shocks)

    for p in res:
        XX  .append(p[0])
        COV .append(p[1])
        EPS .append(p[2])
        PAR .append(p[3])

    EPS = np.array(EPS)
    XX = np.array(XX, dtype=object)
    PAR = np.array(PAR)

    if save is not None:
        smethod = method
        if method is None:
            smethod = -1
        np.savez(save,
                 EPS=EPS,
                 COV=COV,
                 XX=XX,
                 PAR=PAR,
                 OBS_COV=self.obs_cov,
                 method=smethod,
                 presmoothing=presmoothing,
                 is_reduced=reduce_sys
                 )

    self.epstracted = XX, COV, EPS, PAR, self.obs_cov, method

    return XX, COV, EPS, PAR


def sampled_sim(self, epstracted=None, mask=None, reduce_sys=None, forecast=False, linear=False, nr_samples=None, ncores=None, show_warnings=False, verbose=False):

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
