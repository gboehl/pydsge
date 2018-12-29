#!/bin/python2
# -*- coding: utf-8 -*-
import numpy as np
import warnings
import time
import emcee
from tqdm import tqdm
from .stats import InvGamma
import multiprocessing as mp

class modloader(object):
    
    name = 'modloader'

    def __init__(self, filename):

        self.filename   = filename
        self.files      = np.load(filename)
        self.Z          = self.files['Z']
        self.years      = self.files['years']
        self.prior_names    = self.files['prior_names']
        self.chain      = self.files['chain']
        self.prior_dist = self.files['prior_dist']
        self.prior      = self.files['prior_names']
        self.tune       = self.files['tune']
        self.ndraws     = self.files['ndraws']
        self.par_fix    = self.files['par_fix']
        self.prior_arg  = self.files['prior_arg']
        self.modelpath  = str(self.files['modelpath'])

        if 'vv' in self.files:
            self.vv     = self.files['vv']

        print("Results imported. Do not forget to adjust the number of tune-in periods (self.tune).")
    
    def masker(self):
        iss     = np.zeros(len(self.prior_names), dtype=bool)
        for v in self.prior:
            iss = iss | (self.prior_names == v)
        return iss

    def means(self):
        x                   = self.par_fix
        x[self.prior_arg]   = self.chain[:,self.tune:].mean(axis=(0,1))
        return list(x)

    def medians(self):
        x                   = self.par_fix
        x[self.prior_arg]   = np.median(self.chain[:,self.tune:], axis=(0,1))
        return list(x)

    def summary(self):

        from .stats import summary

        return summary(self.chain[:,self.tune:], self.prior_names)

    def traceplot(self, chain=None, varnames=None, tune=None, priors_dist=None, **args):

        from .plots import traceplot

        if chain is None:
            trace_value     = self.chain[:,:,self.masker()]
        else:
            trace_value    = chain
        if varnames is None:
            varnames        = self.prior
        if tune is None:
            tune            = self.tune
        if priors_dist is None:
             priors_dist         = self.prior_dist

        return traceplot(trace_value, varnames=varnames, tune=tune, priors=priors_dist, **args)

    def posteriorplot(self, chain=None, varnames=None, tune=None, **args):

        from .plots import posteriorplot

        if chain is None:
            trace_value     = self.chain[:,:,self.masker()]
        else:
            trace_value     = chain
        if varnames is None:
            varnames        = self.prior
        if tune is None:
            tune            = self.tune

        return posteriorplot(trace_value, varnames=self.prior, tune=self.tune, **args)

    def innovations_mask(self):
        return np.full((self.Z.shape[0]-1, self.Z.shape[1]), np.nan)


def save_res(self, filename):
    np.savez(filename,
             Z              = self.Z,
             vv             = self.vv,
             years          = self.years,
             par_fix        = self.par_fix,
             prior_arg      = self.prior_arg,
             ndraws         = self.ndraws, 
             chain          = self.sampler.chain, 
             prior_dist     = self.sampler.prior_dist, 
             prior_names    = self.sampler.prior_names, 
             tune           = self.sampler.tune, 
             modelpath      = self['filename'],
             means          = self.sampler.par_means)
            

def runner_pooled(nr_samples, ncores, innovations_mask):

    import pathos

    global runner_glob

    def runner_loc(x):
        return runner_glob(x, innovations_mask)

    pool    = pathos.pools.ProcessPool(ncores)

    res     = list(tqdm(pool.imap(runner_loc, range(nr_samples)), unit=' draw(s)', total=nr_samples, dynamic_ncols=True))

    pool.close()
    pool.join()
    pool.clear()

    return res


def sampled_sim(self, be_res = None, alpha = None, innovations_mask = None, nr_samples = 1000, ncores = None):

    import random
    import pathos

    if be_res is None:
        chain   = self.sampler.chain
        tune    = self.sampler.tune
        par_fix = self.par_fix
        prior_arg   = self.prior_arg,

    else:
        chain   = be_res.chain
        tune    = be_res.tune
        par_fix = be_res.par_fix
        prior_arg   = be_res.prior_arg

    if ncores is None:
        ncores    = pathos.multiprocessing.cpu_count()

    all_pars    = chain[:,tune:].reshape(-1,chain.shape[2])

    def runner(nr, innovations_mask):
        random.seed(nr)
        randpar             = par_fix
        randpar[prior_arg]  = random.choice(all_pars)

        self.get_sys(list(randpar), info=False)                      # define parameters
        self.preprocess(info=False)                   # preprocess matrices for speedup

        self.create_filter()
        X, cov      = self.run_filter()
        res         = self.extract()

        if innovations_mask is not None:

            res     = np.where(np.isnan(innovations_mask), res, innovations_mask)

        SZ, SX, SK  = self.simulate(res)

        return SZ, SX, SK

    global runner_glob
    runner_glob    = runner

    res     = runner_pooled(nr_samples, ncores, innovations_mask)

    SZS     = []
    SXS     = []
    SKS     = []

    for p in res:
        SZS.append(p[0])
        SXS.append(p[1])
        SKS.append(p[2])

    return np.array(SZS), np.array(SXS), np.array(SKS)


def sampled_irfs(self, be_res, shocklist, wannasee, nr_samples = 1000, ncores = None):

    import random
    import pathos

    chain   = be_res.chain
    tune    = be_res.tune
    par_fix = be_res.par_fix
    prior_arg   = be_res.prior_arg

    if ncores is None:
        ncores    = pathos.multiprocessing.cpu_count()

    all_pars    = chain[:,tune:].reshape(-1,chain.shape[2])

    ## dry run
    par     = be_res.means()
    self.get_sys(par, info=False)                      # define parameters
    self.preprocess(info=False)                   # preprocess matrices for speedup

    Xlabels = mod.irfs(shocklist, wannasee)[1]

    def runner(nr, useless_arg):
        random.seed(nr)
        randpar             = par_fix
        randpar[prior_arg]  = random.choice(all_pars)

        self.get_sys(list(randpar), info=False)                      # define parameters
        self.preprocess(info=False)                   # preprocess matrices for speedup

        xs, _, (ys, ks, ls)   = mod.irfs(shocklist, wannasee)

        return xs, ys, ks, ls

    global runner_glob
    runner_glob    = runner

    res     = runner_pooled(nr_samples, ncores, None)

    X, Y, K, L  = [], [], [], []

    for p in res:
        X.append(p[0])
        Y.append(p[1])
        K.append(p[2])
        L.append(p[3])

    return np.array(X), Xlabels, (np.array(Y), np.array(K), np.array(L))
