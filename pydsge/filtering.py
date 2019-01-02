#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
from .stuff import *
import pydsge
# from econsieve import UnscentedKalmanFilter as UKF
# from econsieve import ScaledSigmaPoints as SSP
from econsieve import TVF
from econsieve.stats import logpdf
from scipy.optimize import minimize as so_minimize

def create_obs_cov(self, scale_obs = 0.1):

    if not hasattr(self, 'Z'):
        warnings.warn('No time series of observables provided')
    else:
        sig_obs 	= np.var(self.Z, axis = 0)*scale_obs

        self.obs_cov   = np.diagflat(sig_obs)


def create_filter(self, P = None, R = None, N = None):

    np.random.seed(0)

    if not hasattr(self, 'Z'):
        warnings.warn('No time series of observables provided')

    if N is None:
        N   = 5*len(self.vv)

    tvf    = TVF(N, model_obj = self)

    if P is not None:
        tvf.P 		= P
    else:
        tvf.P 		*= 1e1

    if R is not None:
        tvf.R 		= R
    elif hasattr(self, 'obs_cov'):
        tvf.R 		= self.obs_cov

    CO          = self.SIG @ self.QQ(self.par)

    tvf.Q 		= CO @ CO.T

    self.tvf    = tvf


def get_ll(self, info=False):

    if info == 1:
        st  = time.time()

    ll     = self.tvf.batch_filter(self.Z, calc_ll = True, info=info)[2]

    self.ll     = ll

    if info == 1:
        print('Filtering ll done in '+str(np.round(time.time()-st,3))+'seconds.')

    return self.ll


def run_filter(self, use_rts=True, info=False):

    if info == 1:
        st  = time.time()

    if use_rts:
        store   = True
    else:
        store   = False

    X1, cov, ll     = self.tvf.batch_filter(self.Z, store=store, info=info)

    if use_rts:
        X1, cov     = self.tvf.rts_smoother(X1, cov)

    if info == 1:
        print('Filtering done in '+str(np.round(time.time()-st,3))+'seconds.')

    self.filtered_X      = X1
    self.filtered_cov    = cov
    
    return X1, cov


def extract(self, X=None, cov=None, info=True):

    if X is None:
        X   = self.filtered_X

    if cov is None:
        cov = self.filtered_cov


    x   = X[0]
    EPS = []

    flag    = False
    flags   = False

    if info:
        st  = time.time()


    for t in range(X[:-1].shape[0]):

        # mtd     = 'BFGS'   
        mtd     = 'L-BFGS-B' 

        eps0    = np.zeros(len(self.shocks))

        eps2x   = lambda eps: self.t_func(x, noise=eps, return_flag=False)
        target  = lambda eps: -logpdf(eps2x(eps), mean = X[t+1], cov = cov[t+1])

        res     = so_minimize(target, eps0, method = mtd)

        if not res['success']:
            res     = so_minimize(target, eps0, method = 'Powell')

            if not res['success']:
                if flag:
                    flags   = True
                flag    = True

        eps     = res['x']

        EPS.append(eps)
        x   = eps2x(eps)

    if info:
        print('Extraction took ', time.time() - st, 'seconds.')
        if flags:
            warnings.warn('Several issues with convergence.')
        elif flag:
            warnings.warn('Issue with convergence')

    self.res = np.array(EPS)

    return self.res
