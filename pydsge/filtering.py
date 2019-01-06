#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
from .stuff import *
import pydsge
# from econsieve import UnscentedKalmanFilter as UKF
# from econsieve import ScaledSigmaPoints as SSP
from econsieve import EnKF
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

    enkf    = EnKF(N, model_obj = self)

    if P is not None:
        enkf.P 		= P
    else:
        enkf.P 		*= 1e1

    if R is not None:
        enkf.R 		= R
    elif hasattr(self, 'obs_cov'):
        enkf.R 		= self.obs_cov

    CO          = self.SIG @ self.QQ(self.par)

    enkf.Q 		= CO @ CO.T

    self.enkf    = enkf


def get_ll(self, info=False):

    if info == 1:
        st  = time.time()

    ll     = self.enkf.batch_filter(self.Z, calc_ll = True, info=info)[2]

    self.ll     = ll

    if info == 1:
        print('Filtering ll done in '+str(np.round(time.time()-st,3))+'seconds.')

    return self.ll


def run_filter(self, use_rts=True, info=False):

    if info == 1:
        st  = time.time()

    X1, cov, ll     = self.enkf.batch_filter(self.Z, store=use_rts, info=info)

    if use_rts:
        X1, cov     = self.enkf.rts_smoother(X1, cov)

    if info == 1:
        print('Filtering done in '+str(np.round(time.time()-st,3))+'seconds.')

    self.filtered_X      = X1
    self.filtered_cov    = cov
    
    return X1, cov


def extract(self, mean = None, cov = None, mtd = None, info = True):

    if mean is None:
        mean   = self.filtered_X

    if cov is None:
        cov = self.filtered_cov

    if mtd is None:
        mtd     = 'L-BFGS-B' 

    x   = mean[0]
    EPS = []

    flag    = False
    flags   = False

    if info:
        st  = time.time()

    def target(eps, x, mean, cov):

        state, flag     = self.t_func(x, noise=eps)

        if flag:
            return np.inf
        else:
            return -logpdf(state, mean = mean, cov = cov)


    for t in range(mean[:-1].shape[0]):


        eps0    = np.zeros(len(self.shocks))

        res     = so_minimize(target, eps0, method = mtd, args = (x, mean[t+1], cov[t+1]))

        ## backup option
        if not res['success'] and mtd is not 'Powell':
            res     = so_minimize(target, eps0, method = 'Powell', args = (x, mean[t+1], cov[t+1]))

            if not res['success']:
                if flag:
                    flags   = True
                flag    = True

        eps     = res['x']

        EPS.append(eps)

        x   = self.t_func(x, noise=eps)[0]

    if info:
        print('Extraction took ', time.time() - st, 'seconds.')
    if flags:
        warnings.warn('Issues(!) with convergence.')
    elif flag:
        warnings.warn('Issue with convergence')

    self.res = np.array(EPS)

    return self.res
