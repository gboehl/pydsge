#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
from .stuff import *
import pydsge
from econsieve import EnKF
from econsieve import KalmanFilter
from econsieve.stats import logpdf


def create_obs_cov(self, scale_obs=0.1):

    if not hasattr(self, 'Z'):
        warnings.warn('No time series of observables provided')
    else:
        sig_obs = np.var(self.Z, axis=0)*scale_obs**2

        self.obs_cov = np.diagflat(sig_obs)


def create_filter(self, P=None, R=None, N=None, linear=False):

    if N is None:
        N = 500

    np.random.seed(0)

    if not hasattr(self, 'Z'):
        warnings.warn('No time series of observables provided')

    if linear:
        xkf = KalmanFilter(dim_x=len(self.vv), dim_z=self.ny)
        xkf.F = self.linear_representation()
        xkf.H = self.hx
    else:
        xkf = EnKF(N=N, dim_x=len(self.vv), dim_z=self.ny,
                   fx=self.t_func, hx=self.o_func, model_obj=self)

    if P is not None:
        xkf.P = P
    elif hasattr(self, 'P'):
        xkf.P = self.P
    else:
        xkf.P *= 1e1

    if R is not None:
        xkf.R = R
    elif hasattr(self, 'obs_cov'):
        xkf.R = self.obs_cov

    xkf.eps_cov = self.QQ(self.par)

    CO = self.SIG @ xkf.eps_cov

    xkf.Q = CO @ CO.T

    if linear:
        self.kf = xkf
    else:
        self.enkf = xkf


def get_ll(self, verbose=False):

    if verbose:
        st = time.time()

    if hasattr(self, 'enkf'):
        # set approximation to get ll
        self.enkf.fx = lambda x: self.t_func(x)

        ll = self.enkf.batch_filter(self.Z, calc_ll=True, verbose=verbose)
    else:
        ll = self.kf.batch_filter(self.Z)[2]

    if np.isnan(ll):
        ll = -np.inf

    self.ll = ll

    if verbose:
        print('[get_ll:]'.ljust(15, ' ')+'Filtering done in %s seconds. Likelihood is %s.' %
              (np.round(time.time()-st, 3), ll))

    return self.ll


def run_filter(self, use_rts=True, rcond=1e-14, verbose=False):

    if verbose:
        st = time.time()

    if hasattr(self, 'enkf'):
        # set approximation level
        self.enkf.fx = lambda x: self.t_func(x)

        X1, cov = self.enkf.batch_filter(
            self.Z, store=use_rts, verbose=verbose)

        if use_rts:
            X1, cov = self.enkf.rts_smoother(X1, cov, rcond=rcond)
    else:
        X1, cov, ll = self.kf.batch_filter(self.Z)

        if use_rts:
            X1, cov, _, _ = self.kf.rts_smoother(X1, cov)

    if verbose:
        print('[run_filter:]'.ljust(15, ' ')+'Filtering done in ' +
              str(np.round(time.time()-st, 3))+'seconds.')

    self.filtered_X = X1
    self.filtered_cov = cov

    return X1, cov


def extract(self, pmean=None, cov=None, method=None, converged_only=10, return_flag=False, itype=(0, 1), presmoothing=None, min_options=None, show_warnings=True, verbose=True):

    self.enkf.fx = lambda x, noise: self.t_func(x, noise)

    if pmean is None:
        pmean = self.filtered_X.copy()

    if cov is None:
        cov = self.filtered_cov.copy()

    T1 = self.linear_representation()
    T1, T2 = self.hx[0] @ T1, self.hx[0] @ T1 @ self.SIG
    T3 = self.hx[1]
    mod_objs = (T1, T2, T3), self.SIG

    means, cov, res, flag = self.enkf.ipas(pmean, cov, method, converged_only, show_warnings=show_warnings,
                                           itype=itype, presmoothing=presmoothing, objects=mod_objs, min_options=min_options, return_flag=True, verbose=verbose)

    self.res = res

    if return_flag:
        return means, cov, res, flag

    return means, cov, res
