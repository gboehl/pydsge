#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
from .stuff import *
import pydsge
# from econsieve import UnscentedKalmanFilter as UKF
from econsieve import EnsembleKalmanFilter as EnKF
from econsieve import ScaledSigmaPoints as SSP
from scipy.stats import norm 

def create_filter(self, scale_obs = 0.1, N = 100):

    np.random.seed(0)

    dim_v       = len(self.vv)

    if not hasattr(self, 'Z'):
        warnings.warn('No time series of observables provided')
    else:
        sig_obs 	= np.var(self.Z, axis = 0)*scale_obs

    fx      = lambda x: self.t_func(x, return_flag=False)

    enkf    = EnKF(dim_v, self.ny, fx, self.o_func, N)

    enkf.R 		= np.diag(sig_obs)**2

    enkf.R   = np.diagflat(sig_obs)

    CO          = self.SIG @ self.QQ(self.par)

    enkf.Q 		= CO @ CO.T

    enkf.P 		*= 1e1

    self.enkf    = enkf


def get_ll(self, use_rts=False, info=False):

    if info == 1:
        st  = time.time()

    ll     = self.enkf.batch_filter(self.Z)[2]

    self.ll     = ll

    if info == 1:
        print('Filtering done in '+str(np.round(time.time()-st,3))+'seconds.')

    return self.ll


def run_filter(self, use_rts=True, info=False):

    if info == 1:
        st  = time.time()

    X1, cov, ll     = self.enkf.batch_filter(self.Z)

    if use_rts:
        X1, cov     = self.enkf.rts_smoother(X1, cov)

    if info == 1:
        print('Filtering done in '+str(np.round(time.time()-st,3))+'seconds.')

    self.filtered_X      = X1
    self.filtered_cov    = cov
    
    return X1, cov


def extract(self, X=None, cov=None):

    import numdifftools as nd

    if X is None:
        X   = self.filtered_X

    if cov is None:
        cov = self.filtered_cov

    x   = X[0]
    EPS = []

    for t in range(X[:-1].shape[0]):

        eps2x   = lambda eps: self.t_func(x, noise=eps, return_flag=False)
        jac     = nd.Jacobian(eps2x, step=1e-2, order=1)
        eps     = np.zeros(len(self.shocks))

        mu      = X[t+1]
        P_inv   = nl.pinv(cov[t+1])

        i       = 0
        delta   = 1

        while True:

            G   = jac(eps)
            increment   = nl.pinv(G.T @ P_inv @ G) @ G.T @ P_inv @ (eps2x(eps) - X[t+1])
            eps         -= delta*increment

            fit         = np.max(np.abs(delta*increment)) 

            if fit < 1e-4:
                break

            i   += 1
            if i > 40:
                delta   /= 2
                i       = 0
                if delta < 1e-3:
                    print('...did not converge.')
                    break

        EPS.append(eps)
        x   = eps2x(eps)

    self.res = np.array(EPS)

    return np.array(EPS)
