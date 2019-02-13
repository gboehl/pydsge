#!/bin/python2
# -*- coding: utf-8 -*-
import warnings
import time
import emcee
import pathos
import os.path
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from .stats import InvGamma


class modloader(object):

    name = 'modloader'

    def __init__(self, filename):

        self.filename = filename
        self.files = np.load(filename)
        self.Z = self.files['Z']
        if 'obs_cov' in self.files.files:
            self.obs_cov = self.files['obs_cov']
        if 'init_cov' in self.files.files:
            self.init_cov = self.files['init_cov']
        if 'description' in self.files.files:
            self.description = self.files['description']
        if 'priors' in self.files.files:
            self.priors = self.files['priors'].item()
        if 'acc_frac' in self.files.files:
            self.acc_frac = self.files['acc_frac']
        self.years = self.files['years']
        self.chain = self.files['chain']
        self.prior_names = self.files['prior_names']
        self.prior_dist = self.files['prior_dist']
        self.prior_arg = self.files['prior_arg']
        self.tune = self.files['tune']
        self.ndraws = self.files['ndraws']
        self.par_fix = self.files['par_fix']
        self.modelpath = str(self.files['modelpath'])

        if 'vv' in self.files:
            self.vv = self.files['vv']

        print('[modloader:]'.ljust(
            15, ' ')+'Results imported. Number of burn-in periods is %s out of %s' % (self.tune, self.ndraws))
        print('[modloader:]'.ljust(15, ' ') +
              'Description: '+str(self.description))

    def chain_masker(self):
        iss = np.zeros(len(self.prior_names), dtype=bool)
        for v in self.prior_names:
            iss = iss | (self.prior_names == v)
        return iss

    def means(self):
        x = self.par_fix

        cshp = self.chain.shape
        sum_chain = self.chain.reshape(-1, cshp[-2], cshp[-1])

        x[self.prior_arg] = sum_chain[:, self.tune:].mean(axis=(0, 1))
        return list(x)

    def medians(self):
        x = self.par_fix
        x[self.prior_arg] = np.median(self.chain[:, self.tune:], axis=(0, 1))
        return list(x)

    def summary(self):

        from .stats import summary

        if not hasattr(self, 'priors'):
            self_priors = None
        else:
            self_priors = self.priors

        chain = self.chain.reshape(-1, *self.chain.shape[-2:])

        return summary(chain[:, self.tune:], self_priors)

    def traceplot(self, chain=None, varnames=None, tune=None, priors_dist=None, draw_lines=None, **args):

        from .plots import traceplot

        if chain is None:
            trace_value = self.chain[..., self.chain_masker()]
        else:
            trace_value = chain
        if varnames is None:
            varnames = self.prior_names
        if tune is None:
            tune = self.tune
        if priors_dist is None:
            priors_dist = self.prior_dist

        if draw_lines is None:
            if trace_value.ndim == 4:
                draw_lines = True
            else:
                draw_lines = False

        return traceplot(trace_value, varnames=varnames, tune=tune, priors=priors_dist, draw_lines=draw_lines, **args)

    def posteriorplot(self, chain=None, varnames=None, tune=None, **args):

        from .plots import posteriorplot

        if chain is None:
            trace_raw = self.chain[..., self.chain_masker()]
        else:
            trace_raw = chain
        if varnames is None:
            varnames = self.prior_names
        if tune is None:
            tune = self.tune

        trace_value = trace_raw.reshape(-1, *trace_raw.shape[-2:])

        return posteriorplot(trace_value, varnames=self.prior_names, tune=self.tune, **args)


def mask(self, verbose=False):
    if verbose:
        print('[mask:]'.ljust(15, ' ') + 'Shocks:', self.shocks)
    return np.full((self.Z.shape[0]-1, self.Z.shape[1]), np.nan)


def save_res(self, filename, description=None):

    if not hasattr(self, 'description'):

        self.description = description

        if description is None:
            self.description = ''

    if hasattr(self, 'kf'):
        init_cov = self.kf.P
    else:
        init_cov = self.enkf.P

    np.savez_compressed(filename,
                        Z=self.Z,
                        vv=self.vv,
                        years=self.years,
                        description=self.description,
                        obs_cov=self.obs_cov,
                        init_cov=init_cov,
                        par_fix=self.par_fix,
                        ndraws=self.ndraws,
                        chain=self.sampler.chain,
                        acc_frac=self.sampler.acceptance_fraction,
                        prior_dist=self.sampler.prior_dist,
                        prior_names=self.sampler.prior_names,
                        prior_arg=self.prior_arg,
                        priors=self['__data__']['estimation']['prior'],
                        tune=self.sampler.tune,
                        modelpath=self['filename'],
                        means=self.sampler.par_means)
    print('[save_res:]'.ljust(15, ' ')+'Results saved in ', filename)


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


def posterior_sample(self, be_res=None, seed=0):

    import random

    if be_res is None:
        chain = self.sampler.chain
        tune = self.sampler.tune
        par_fix = self.par_fix
        prior_arg = self.prior_arg,

    else:
        chain = be_res.chain
        tune = be_res.tune
        par_fix = be_res.par_fix
        prior_arg = be_res.prior_arg

    all_pars = chain[..., tune:, :].reshape(-1, chain.shape[-1])

    random.seed(seed)
    randpar = par_fix
    randpar[prior_arg] = random.choice(all_pars)

    return list(randpar)


def epstract(self, be_res=None, N=None, nr_samples=100, save=None, ncores=None, method=None, itype=(0, 1), converged_only=True, max_attempts=3, presmoothing=None, min_options=None, force=False, verbose=False):

    XX = []
    COV = []
    EPS = []
    PAR = []

    if N is None:
        N = 500

    if not force and save is not None and os.path.isfile(save):

        files = np.load(save)
        OBS_COV = files['OBS_COV']
        smethod = files['method']

        c0 = method == smethod
        c1 = method is None and smethod == -1

        EPS = files['EPS']
        COV = files['COV']
        XX = files['XX']
        PAR = files['PAR']

        if EPS.shape[0] >= nr_samples:
            print('[epstract:]'.ljust(15, ' ') +
                  'Epstract already exists')

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
            self.get_sys(par, verbose=False)
            # preprocess matrices for speedup
            self.preprocess(verbose=False)

            self.create_filter(N=N)
            SX, scov = self.run_filter()
            IX, icov, eps, flag = self.extract(method=method, verbose=verbose, converged_only=converged_only,
                                               itype=itype, presmoothing=presmoothing, min_options=min_options, return_flag=True)

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
                 method=smethod
                 )

    self.epstracted = XX, COV, EPS, PAR, self.obs_cov, method

    return XX, COV, EPS, PAR


def sampled_sim(self, epstracted=None, mask=None, forecast=False, linear=False, nr_samples=None, ncores=None, show_warnings=False, verbose=False):

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    if epstracted is None:
        epstracted = self.epstracted[:4]

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

        self.get_sys(par, verbose=verbose)
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


def sampled_irfs(self, be_res, shocklist, wannasee, nr_samples=1000, ncores=None, show_warnings=False):

    import pathos

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    # dry run
    par = be_res.means()
    # define parameters
    self.get_sys(par, 'all', verbose=False)
    # preprocess matrices for speedup
    self.preprocess(verbose=False)

    Xlabels = self.irfs(shocklist, wannasee)[1]

    def runner(nr, useless_arg):

        par = self.posterior_sample(be_res=be_res, seed=nr)

        # define parameters
        self.get_sys(par, 'all', verbose=False)
        # preprocess matrices for speedup
        self.preprocess(verbose=False)

        xs, _, (ys, ks, ls) = self.irfs(
            shocklist, wannasee, show_warnings=show_warnings)

        return xs, ys, ks, ls

    global runner_glob
    runner_glob = runner

    res = runner_pooled(nr_samples, ncores, None)

    X, Y, K, L = [], [], [], []

    for p in res:
        X.append(p[0])
        Y.append(p[1])
        K.append(p[2])
        L.append(p[3])

    return np.array(X), Xlabels, (np.array(Y), np.array(K), np.array(L))
