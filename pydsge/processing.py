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
        if 'obs_cov' in self.files.files:
            self.obs_cov        = self.files['obs_cov']
        if 'description' in self.files.files:
            self.description    = self.files['description']
        if 'priors' in self.files.files:
            self.priors         = self.files['priors'].item()
        self.years      = self.files['years']
        self.chain      = self.files['chain']
        self.prior_names    = self.files['prior_names']
        self.prior_dist = self.files['prior_dist']
        self.prior_arg  = self.files['prior_arg']
        self.tune       = self.files['tune']
        self.ndraws     = self.files['ndraws']
        self.par_fix    = self.files['par_fix']
        self.modelpath  = str(self.files['modelpath'])

        if 'vv' in self.files:
            self.vv     = self.files['vv']

        print("Results imported. Number of burn-in periods is %s out of %s" %(self.tune, self.ndraws))
    
    def masker(self):
        iss     = np.zeros(len(self.prior_names), dtype=bool)
        for v in self.prior_names:
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

        if not hasattr(self, 'priors'):
            self_priors     = None
        else:
            self_priors     = self.priors

        return summary(self.chain[:,self.tune:], self_priors)

    def traceplot(self, chain=None, varnames=None, tune=None, priors_dist=None, **args):

        from .plots import traceplot

        if chain is None:
            trace_value     = self.chain[:,:,self.masker()]
        else:
            trace_value    = chain
        if varnames is None:
            varnames        = self.prior_names
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
            varnames        = self.prior_names
        if tune is None:
            tune            = self.tune

        return posteriorplot(trace_value, varnames=self.prior_names, tune=self.tune, **args)

    def innovations_mask(self):
        return np.full((self.Z.shape[0]-1, self.Z.shape[1]), np.nan)


def save_res(self, filename, description = ''):
    np.savez(filename,
             Z              = self.Z,
             vv             = self.vv,
             years          = self.years,
             description    = description,
             obs_cov        = self.obs_cov,
             par_fix        = self.par_fix,
             ndraws         = self.ndraws, 
             chain          = self.sampler.chain, 
             prior_dist     = self.sampler.prior_dist, 
             prior_names    = self.sampler.prior_names, 
             prior_arg      = self.prior_arg,
             priors         = self['__data__']['estimation']['prior'],
             tune           = self.sampler.tune, 
             modelpath      = self['filename'],
             means          = self.sampler.par_means)
    print('Results saved in ', filename)
            

def runner_pooled(nr_samples, ncores, innovations_mask):

    import pathos

    global runner_glob

    def runner_loc(x):
        return runner_glob(x, innovations_mask)

    pool    = pathos.pools.ProcessPool(ncores)

    res     = list(tqdm(pool.uimap(runner_loc, range(nr_samples)), unit=' draw(s)', total=nr_samples, dynamic_ncols=True))

    pool.close()
    pool.join()
    pool.clear()

    return res

def posterior_sample(self, be_res = None, seed = 0):

    import random

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

    all_pars    = chain[:,tune:].reshape(-1,chain.shape[2])

    random.seed(seed)
    randpar             = par_fix
    randpar[prior_arg]  = random.choice(all_pars)

    return list(randpar)


def epstract(self, be_res = None, nr_samples = 1000, save = None, ncores = None, method = None, converged_only = True, info = False):

    EPS     = []
    X0      = []
    PAR     = []

    if save is not None:

        import os.path

        if os.path.isfile(save):

            files   = np.load(save)
            EPS     = files['EPS']
            X0      = files['X0']
            PAR     = files['PAR']
            COV     = files['COV']
            smethod = files['method']

            c0  = method == smethod 
            c1  = method is None and smethod == -1

            if np.all(COV == self.obs_cov) and (c0 or c1):
                if EPS.shape[0] >= nr_samples:
                    print('epstract: Epstract already exists')

                    self.epstracted     = EPS, X0, PAR, self.obs_cov, method

                    return EPS, X0, PAR

                else:
                    print('Appending to existing epstract...')

                    EPS     = list(EPS)
                    X0      = list(X0)
                    PAR     = list(PAR)

    import pathos

    if ncores is None:
        ncores    = pathos.multiprocessing.cpu_count()

    yet         = len(EPS)
    nr_samples  = nr_samples - yet

    def runner(nr, innovations_mask):

        par     = self.posterior_sample(be_res = be_res, seed = yet+nr)

        self.get_sys(par, info=False)                      # define parameters
        self.preprocess(info=False)                   # preprocess matrices for speedup

        self.create_filter()
        X, cov      = self.run_filter()
        eps         = self.extract(method = method, info = info, converged_only = converged_only)[2]

        return eps, X[0], par

    global runner_glob
    runner_glob    = runner

    res     = runner_pooled(nr_samples, ncores, None)

    no_obs, dim_z   = self.Z.shape
    dim_e           = len(self.shocks)

    for p in res:
        EPS .append(p[0])
        X0  .append(p[1])
        PAR .append(p[2])

    EPS     = np.array(EPS)
    X0      = np.array(X0, dtype=object)
    PAR     = np.array(PAR)

    if save is not None:
        smethod     = method
        if method is None:
            smethod     = -1
        np.savez(save,
                 EPS    = EPS,
                 X0     = X0,
                 PAR    = PAR,
                 COV    = self.obs_cov,
                 method = smethod
                )

    self.epstracted     = EPS, X0, PAR, self.obs_cov, method

    return EPS, X0, PAR


def sampled_sim(self, innovations_mask = None, nr_samples = None, epstracted = None, ncores = None, show_warnings = False, info = False):

    import pathos

    if epstracted is None:
        epstracted  = self.epstracted[:3]

    if ncores is None:
        ncores  = pathos.multiprocessing.cpu_count()

    EPS, X0, PAR    = epstracted     

    if nr_samples is None:
        nr_samples  = EPS.shape[0]

    def runner(nr, innovations_mask):

        par     = list(PAR[nr])
        eps     = EPS[nr]
        x0      = X0[nr]
        
        self.get_sys(par, info=False)                      # define parameters
        self.preprocess(info=False)                   # preprocess matrices for speedup


        if innovations_mask is not None:
            eps     = np.where(np.isnan(innovations_mask), eps, innovations_mask)

        SZ, SX, SK  = self.simulate(eps, initial_state = x0, show_warnings = show_warnings)

        return SZ, SX, SK, self.vv

    global runner_glob
    runner_glob    = runner

    res     = runner_pooled(nr_samples, ncores, innovations_mask)

    dim_x   = 1e50
    for p in res:
        if len(p[3]) < dim_x:
            minr    = p[3]
            dim_x   = len(minr)

    no_obs, dim_z   = self.Z.shape
    dim_e           = len(self.shocks)

    SZS     = np.empty((nr_samples, no_obs, dim_z))
    SXS     = np.empty((nr_samples, no_obs, dim_x))
    SKS     = np.empty((nr_samples, no_obs, 1))

    for n, p in enumerate(res):

        if len(p[3]) > dim_x:
            idx     = [ list(p[3]).index(v) for v in minr ]
            SXS[n,:]  = p[1][:,idx]
        else:
            SXS[n,:]  = p[1]

        SZS[n,:]  = p[0]
        SKS[n,:]  = p[2]

    return SZS, SXS, SKS


def sampled_irfs(self, be_res, shocklist, wannasee, nr_samples = 1000, ncores = None, show_warnings = False):

    import pathos

    if ncores is None:
        ncores    = pathos.multiprocessing.cpu_count()

    ## dry run
    par     = be_res.means()
    self.get_sys(par, 'all', info=False)                      # define parameters
    self.preprocess(info=False)                   # preprocess matrices for speedup

    Xlabels = self.irfs(shocklist, wannasee)[1]

    def runner(nr, useless_arg):

        par     = self.posterior_sample(be_res = be_res, seed = nr)

        self.get_sys(par, 'all', info=False)                      # define parameters
        self.preprocess(info=False)                   # preprocess matrices for speedup

        xs, _, (ys, ks, ls)   = self.irfs(shocklist, wannasee, show_warnings = show_warnings)

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

