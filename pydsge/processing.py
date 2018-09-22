#!/bin/python2
# -*- coding: utf-8 -*-
import numpy as np
import warnings
import time
import emcee
from tqdm import tqdm
from .stats import InvGamma
import multiprocessing as mp
import ctypes 

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
             years          = self.years,
             par_fix        = self.par_fix,
             prior_arg      = self.prior_arg,
             ndraws         = self.ndraws, 
             chain          = self.sampler.chain, 
             prior_dist     = self.sampler.prior_dist, 
             prior_names    = self.sampler.prior_names, 
             tune           = self.sampler.tune, 
             means          = self.sampler.par_means)

def runner_pooled(nr_samples, nr_cores):

    global runner_glob

    def runner_loc(x):

        return runner_glob(x)

    import pathos
    pool    = pathos.pools.ProcessPool(nr_cores)

    res     = list(tqdm(pool.imap(runner_loc, range(nr_samples)), unit=' draw(s)', total=nr_samples, dynamic_ncols=True))

    return res

def sampled_sim(self, be_res, innovations_mask, nr_samples = 1000, nr_cores = None):

    import random

    if nr_cores is None:
        nr_cores   = mp.cpu_count()

    all_pars    = be_res.chain[:,be_res.tune:].reshape(-1,be_res.chain.shape[2])


    def runner(nr):

        randpar                 = be_res.par_fix
        randpar[be_res.prior_arg]  = random.choice(all_pars)

        self.get_sys(list(randpar), info=False)                      # define parameters
        self.preprocess(info=False)                   # preprocess matrices for speedup

        self.create_filter()

        EPS     = self.run_filter(use_rts=True, info=False)[2]

        EPS     = np.where(np.isnan(innovations_mask), EPS, innovations_mask)

        SZ, SX      = self.simulate(EPS)

        return SZ, SX

    global runner_glob
    runner_glob    = runner

    res     = runner_pooled(nr_samples, nr_cores)

    SZS     = []
    SXS     = []

    for p in res:
        SZS.append(p[0])
        SXS.append(p[1])

    return np.array(SZS), np.array(SXS)

