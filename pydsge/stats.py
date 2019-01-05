#!/bin/python2
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def mc_error(x):
    means   = np.mean(x,0)
    return np.std(means) / np.sqrt(x.shape[0])

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width

    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]

    return hdi_min, hdi_max


def _hpd_df(x, alpha):

    cnames = ['hpd_{0:g}'.format(100 * alpha / 2),
              'hpd_{0:g}'.format(100 * (1 - alpha / 2))]

    sx          = np.sort(x.flatten())
    hpd_vals    = np.array(calc_min_interval(sx, alpha)).reshape(1,-1)

    return pd.DataFrame(hpd_vals, columns=cnames)

def summary(trace, varnames, priors, alpha=0.05):
    ## in parts stolen from pymc3 because it looks really nice

    f_prs = [lambda x: pd.Series(x, name='distribution'),
             lambda x: pd.Series(x, name='mean/alpha'),
             lambda x: pd.Series(x, name='sd/beta')]

    funcs = [lambda x: pd.Series(np.mean(x), name='mean'),
             lambda x: pd.Series(np.std(x), name='sd'),
             lambda x: pd.Series(mc_error(x), name='mc_error'),
             lambda x: _hpd_df(x, alpha)]


    var_dfs = []
    for i, var in enumerate(varnames):
        lst     = []
        vals    = trace[:,:,i]

        if priors is not None:
            prior   = priors[var]
            [lst.append(f(prior[j])) for j,f in enumerate(f_prs)]
            
        [lst.append(f(vals)) for f in funcs]
        var_df  = pd.concat(lst, axis=1)
        var_df.index    = [var]
        var_dfs.append(var_df)

    dforg = pd.concat(var_dfs, axis=0)

    return dforg


def mc_mean(trace, varnames):
    ## in most parts just stolen from pymc3 because it looks really nice

    p_means     = []

    for i, var in enumerate(varnames):
        vals = trace[:,:,i]
        p_means.append(np.mean(vals))

    return p_means

class InvGamma(object):
    
    name = 'inv_gamma'

    def __init__(self, a, b):

        self.a = a
        self.b = b

    def logpdf(self, x):

        from scipy.special import gammaln

        a = self.a
        b = self.b
        
        lpdf    = np.copy(x)

        lpdf[x < 0]     = -np.inf

        lpdf[x >= 0]    = (np.log(2) - gammaln(b/2) + b/2*np.log(b*a**2/2)
                -(b+1)/2*np.log(x[ x>=0 ]**2) - b*a**2/(2*x**2))

        return lpdf
