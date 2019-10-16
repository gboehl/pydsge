#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from .stuff import time
from grgrlib.stuff import timeprint
from econsieve.stats import logpdf


def create_obs_cov(self, scale_obs=0.1):

    self.Z = np.array(self.data)
    sig_obs = np.var(self.Z, axis=0)*scale_obs**2
    obs_cov = np.diagflat(sig_obs)

    return obs_cov


def create_filter(self, P=None, R=None, N=None, ftype=None, random_seed=None):

    self.Z = np.array(self.data)

    if random_seed is not None:
        np.random.seed(random_seed)

    if ftype == 'KalmanFilter' or ftype == 'KF':

        from econsieve import KalmanFilter

        f = KalmanFilter(dim_x=len(self.vv), dim_z=self.ny)
        f.F = self.linear_representation
        f.H = self.hx

    elif ftype in ('PF', 'APF', 'ParticleFilter', 'AuxiliaryParticleFilter'):

        from .partfilt import ParticleFilter

        if N is None:
            N = 10000

        aux_bs = ftype in ('AuxiliaryParticleFilter', 'APF')
        f = ParticleFilter(N=N, dim_x=len(self.vv),
                           dim_z=self.ny, auxiliary_bootstrap=aux_bs)

    else:

        from econsieve import TEnKF

        if N is None:
            N = 500
        f = TEnKF(N=N, dim_x=len(self.vv), dim_z=self.ny)

    if P is not None:
        f.P = P
    elif hasattr(self, 'P'):
        f.P = self.P
    else:
        f.P *= 1e1
    f.init_P = f.P

    if R is not None:
        f.R = R

    f.eps_cov = self.QQ(self.par)
    f.Q = self.QQ(self.par) @ self.QQ(self.par)

    if ftype in ('KalmanFilter', 'KF'):
        CO = self.SIG @ f.eps_cov
        f.Q = CO @ CO.T

    self.filter = f

    return f


def get_ll(self, **args):
    return run_filter(self, smoother=False, get_ll=True, **args)


def run_filter(self, smoother=True, get_ll=False, dispatch=None, rcond=1e-14, constr_data=None, verbose=False):

    if verbose:
        st = time.time()

    if constr_data is None:
        if self.filter.name == 'ParticleFilter':
            constr_data = 'elb_level'  # wild guess
        else:
            constr_data = False

    if constr_data:
        # copy the data
        data = self.data
        # constaint const_obs
        x_shift = self.get_calib(constr_data)
        data[str(self.const_obs)] = np.maximum(
            data[str(self.const_obs)], x_shift)
        # send to filter
        self.Z = np.array(data)
    else:
        self.Z = np.array(self.data)

    if dispatch is None:
        dispatch = self.filter.name == 'ParticleFilter'

    if dispatch:
        t_func_jit, o_func_jit, get_eps_jit = self.func_dispatch(full=True)
        self.filter.t_func = t_func_jit
        self.filter.o_func = o_func_jit
        self.filter.get_eps = get_eps_jit

    else:
        self.filter.t_func = self.t_func
        self.filter.o_func = self.o_func
        self.filter.get_eps = self.get_eps

    if self.filter.name == 'KalmanFilter':

        res, cov, ll = self.filter.batch_filter(self.Z)

        if get_ll:
            res = ll

        if smoother:
            res, cov, _, _ = self.filter.rts_smoother(res, cov)

        self.cov = cov

    elif self.filter.name == 'ParticleFilter':

        res = self.filter.batch_filter(self.Z)

        if smoother:

            if verbose:
                print('[run_filter:]'.ljust(
                    15, ' ')+'Filtering done after %s seconds, starting smoothing...' % np.round(time.time()-st, 3))

            if isinstance(smoother, bool):
                smoother = 10
            res = self.filter.smoother(smoother)

    else:

        res = self.filter.batch_filter(
            self.Z, calc_ll=get_ll, store=smoother, verbose=verbose)

        if smoother:
            res = self.filter.rts_smoother(res, rcond=rcond)

    if get_ll:
        if np.isnan(res):
            res = -np.inf
        self.ll = res

        if verbose:
            print('[run_filter:]'.ljust(15, ' ')+'Filtering done in %s. Likelihood is %s.' %
                  (timeprint(time.time()-st, 3), res))
    else:
        self.X = res

        if verbose:
            print('[run_filter:]'.ljust(15, ' ')+'Filtering done in %s.' %
                  timeprint(time.time()-st, 3))

    return res


def extract(self, precalc=True, verbose=True, **npasargs):

    if precalc:
        get_eps = self.filter.get_eps
    else:
        get_eps = None

    means, cov, res, flag = self.filter.npas(
        get_eps=get_eps, verbose=verbose, **npasargs)

    self.means = means
    self.cov = cov
    self.res = res

    return means, cov, res, flag
