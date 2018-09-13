#!/bin/python2
# -*- coding: utf-8 -*-
import numpy as np
from .base import *
from .pyzlb import boehlgorithm
import pydsge

def t_func(self, state, noise = None, return_k = False):

    if noise is not None:
        state   += self.SIG @ noise
    newstate, (l,k), flag   = boehlgorithm(self, state)

    if return_k: 	return newstate, (l,k), flag
    else: 			return newstate

pydsge.DSGE.DSGE.t_func             = t_func

def create_filter(self, alpha = .2, scale_obs = .2):

    from filterpy_dsge.kalman import UnscentedKalmanFilter as UKF
    # from filterpy.kalman import ReducedScaledSigmaPoints
    # from filterpy.kalman import MerweScaledSigmaPoints
    from filterpy_dsge.kalman import SigmaPoints_ftl

    dim_v       = len(self.vv)
    beta_ukf 	= 2.
    kappa_ukf 	= 3 - dim_v

    if not hasattr(self, 'Z'):
        warnings.warn('No time series of observables provided')
    else:
        sig_obs 	= np.std(self.Z, 0)*scale_obs

    exo_args    = ~fast0(self.SIG,1)

    ## ReducedScaledSigmaPoints are an attemp to reduce the number of necessary sigma points. 
    ## As of yet not functional
    # spoints     = ReducedScaledSigmaPoints(alpha, beta_ukf, kappa_ukf, exo_args)

    spoints     = SigmaPoints_ftl(dim_v,alpha, beta_ukf, kappa_ukf)
    # ukf 		= UKF(dim_x=dim_v, dim_z=self.ny, hx=self.obs_arg, fx=self.t_func, points=spoints)
    ukf 		= UKF(dim_x=dim_v, dim_z=self.ny, hx=self.hx, fx=self.t_func, points=spoints)
    ukf.x 		= np.zeros(dim_v)
    ukf.R 		= np.diag(sig_obs)**2

    CO          = self.SIG @ self.QQ(self.par)

    ukf.Q 		= CO @ CO.T

    self.ukf    = ukf


def run_filter(self, use_rts=False, info=False):

    if info == 1:
        st  = time.time()

    exo_args    = ~fast0(self.SIG,1)

    X1, cov, Y, ll     = self.ukf.batch_filter(self.Z)

    ## the actual filter seems to work better than the smoother. The implemented version (succesfully) 
    ## uses the pseudoinverse to deal with the fact that the co-variance matrix is singular
    if use_rts:
        X1, _, _            = self.ukf.rts_smoother(X1, cov)

    # self.filtered_Z     = X1[:,self.obs_arg]
    self.filtered_Z     = (self.hx[0] @ X1.T).T + self.hx[1]
    self.filtered_X     = X1
    self.filtered_V     = X1[:,exo_args]
    self.ll             = ll
    self.residuals      = Y[:,exo_args]
    # self.residuals      = Y

    if info == 1:
        print('Filtering done in '+str(np.round(time.time()-st,3))+'seconds.')
