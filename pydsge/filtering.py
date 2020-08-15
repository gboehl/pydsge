#!/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from grgrlib.core import timeprint
from econsieve.stats import logpdf


def create_obs_cov(self, scale_obs=0.1):

    self.Z = np.array(self.data)
    sig_obs = np.var(self.Z, axis=0)*scale_obs**2
    obs_cov = np.diagflat(sig_obs)

    return obs_cov


def create_filter(self, P=None, R=None, N=None, ftype=None, x_space=False, seed=None, **fargs):

    self.Z = np.array(self.data)
    dim = self.dimq if x_space else self.nvar 

    if ftype == 'KalmanFilter':
        ftype = 'KF'
    if ftype == 'ParticleFilter':
        ftype = 'PF'
    if ftype == 'AuxiliaryParticleFilter':
        ftype = 'APF'
    if ftype == 'KF':

        from econsieve import KalmanFilter

        f = KalmanFilter(dim_x=self.nvar+self.neps, dim_z=self.nobs)

    elif ftype in ('PF', 'APF'):

        print(
            'Warning: Particle filter is experimental and currently not under development.')
        from .pfilter import ParticleFilter

        if N is None:
            N = 10000

        aux_bs = ftype == 'APF'
        f = ParticleFilter(N=N, dim_x=self.nvar,
                           dim_z=self.nobs, auxiliary_bootstrap=aux_bs)

    else:
        ftype = 'TEnKF'

        from econsieve import TEnKF

        if N is None:
            N = 500
        f = TEnKF(N=N, dim_x=dim, dim_z=self.nobs, seed=seed, **fargs)

    if P is not None:
        f.P = P
    elif hasattr(self, 'P'):
        f.P = self.P
    else:
        f.P *= 1e1
    f.init_P = f.P

    if R is not None:
        f.R = R

    f.Q = self.QQ(self.ppar) @ self.QQ(self.ppar)
    f.x_space = x_space
    self.filter = f

    return f


def get_ll(self, **args):
    return run_filter(self, smoother=False, get_ll=True, **args)


def run_filter(self, smoother=True, get_ll=False, dispatch=None, rcond=1e-14, verbose=False):

    if verbose:
        st = time.time()

    self.Z = np.array(self.data)

    # assign latest transition & observation functions (of parameters)
    if self.filter.name == 'KalmanFilter':
        F, E = self.lin_sys
        EE = np.vstack((E, np.eye(self.neps)))

        self.filter.F = np.pad(F, ((0, self.neps), (0, self.neps)))
        self.filter.H = np.pad(
            self.hy[0], ((0, 0), (0, self.neps))), self.hy[1]
        self.filter.Q = EE @ self.filter.Q @ EE.T

    elif dispatch or self.filter.name == 'ParticleFilter':
        from .engine import func_dispatch
        t_func_jit, o_func_jit, get_eps_jit = func_dispatch(self, full=True)
        self.filter.t_func = t_func_jit
        self.filter.o_func = o_func_jit
        self.filter.get_eps = get_eps_jit

    else:
        if self.filter.x_space:
            self.filter.t_func = lambda *x: self.t_func(*x, x_space=self.filter.x_space)
            self.filter.o_func = None
        else:
            self.filter.t_func = self.t_func
            self.filter.o_func = self.o_func
    # self.filter.get_eps = self.get_eps_lin

    if self.filter.name == 'KalmanFilter':

        means, covs, ll = self.filter.batch_filter(self.Z)

        if smoother:
            means, covs, _, _ = self.filter.rts_smoother(
                means, covs, inv=np.linalg.pinv)

        if get_ll:
            res = ll
        else:
            eps = means[:, -self.neps:]
            means = means[:, :-self.neps]
            res = (means, covs, eps)

    elif self.filter.name == 'ParticleFilter':

        res = self.filter.batch_filter(self.Z)

        if smoother:

            if verbose > 0:
                print('[run_filter:]'.ljust(
                    15, ' ')+' Filtering done after %s seconds, starting smoothing...' % np.round(time.time()-st, 3))

            if isinstance(smoother, bool):
                smoother = 10
            res = self.filter.smoother(smoother)

    else:

        res = self.filter.batch_filter(
            self.Z, calc_ll=get_ll, store=smoother, verbose=verbose > 0)

        if smoother:
            res = self.filter.rts_smoother(res, rcond=rcond)

    if get_ll:
        if np.isnan(res):
            res = -np.inf
        self.ll = res
    else:
        self.X = res

    if verbose > 0:
        mess = '[run_filter:]'.ljust(
            15, ' ')+' Filtering done in %s.' % timeprint(time.time()-st, 3)
        if get_ll:
            mess += 'Likelihood is %s.' % res
        print(mess)

    return res


def extract(self, sample=None, nsamples=1, precalc=True, seed=0, nattemps=4, verbose=True, debug=False, l_max=None, k_max=None, **npasargs):
    """Extract the timeseries of (smoothed) shocks.

    Parameters
    ----------
    sample : array, optional
        Provide one or several parameter vectors used for which the smoothed shocks are calculated (default is the current `self.par`)
    nsamples : int, optional
        Number of `npas`-draws for each element in `sample`. Defaults to 1
    nattemps : int, optional
        Number of attemps per sample to crunch the sample with a different seed. Defaults to 4

    Returns
    -------
    tuple
        The result(s)
    """

    import tqdm
    import os
    from grgrlib.core import map2arr, serializer

    if sample is None:
        sample = self.par

    if np.ndim(sample) <= 1:
        sample = [sample]

    fname = self.filter.name
    verbose = max(verbose, 9*debug)

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    if fname == 'ParticleFilter':
        raise NotImplementedError

    elif fname == 'KalmanFilter':
        if nsamples > 1:
            print('[extract:]'.ljust(
                15, ' ')+' Setting `nsamples` to 1 as the linear filter is deterministic.')
        nsamples = 1
        debug = not hasattr(self, 'debug') or self.debug
        self.debug = True

    else:
        npas = serializer(self.filter.npas)

    if self.filter.dim_x != self.nvar:
        raise RuntimeError(
            'Shape mismatch between dimensionality of filter and model.')

    else:
        self.debug |= debug

    set_par = serializer(self.set_par)
    run_filter = serializer(self.run_filter)
    t_func = serializer(self.t_func)
    obs = serializer(self.obs)
    # filter_get_eps = serializer(self.get_eps_lin)
    edim = len(self.shocks)

    sample = [(x, y) for x in sample for y in range(nsamples)]

    def runner(arg):

        par, seed_loc = arg

        if par is not None:
            set_par(par, l_max=l_max, k_max=k_max)

        res = run_filter(verbose=verbose > 2)

        if fname == 'KalmanFilter':
            raise NotImplementedError(
                'extraction for linear KF is rewritten via KF smoother. To be re-implemented.')
            means, covs = res
            res = means.copy()
            resid = np.empty((means.shape[0]-1, edim))

            for t, x in enumerate(means[1:]):
                resid[t] = filter_get_eps(x, res[t])
                res[t+1] = t_func(res[t], resid[t], linear=True)[0]

            return res, obs(res), covs, resid, 0

        # get_eps = filter_get_eps if precalc else None

        for natt in range(nattemps):
            np.random.seed(seed_loc)
            seed_loc = np.random.randint(2**31)  # win explodes with 2**32
            try:
                means, covs, resid, flags = npas(
                    get_eps=None, verbose=max(len(sample) == 1, verbose-1), seed=seed_loc, nsamples=1, **npasargs)

                return means[0], obs(means[0]), covs, resid[0], flags
            except Exception as e:
                ee = e

        import sys
        raise type(ee)(str(ee) + ' (after %s unsuccessful attemps).' %
                       (natt+1)).with_traceback(sys.exc_info()[2])

    wrap = tqdm.tqdm if (verbose and len(sample) >
                         1) else (lambda x, **kwarg: x)
    res = wrap(self.mapper(runner, sample), unit=' sample(s)',
               total=len(sample), dynamic_ncols=True)
    means, obs, covs, resid, flags = map2arr(res)

    if hasattr(self, 'pool') and self.pool:
        self.pool.close()

    if fname == 'KalmanFilter':
        self.debug = debug

    if means.shape[0] == 1:
        means = pd.DataFrame(means[0], index=self.data.index, columns=self.vv)
        resid = pd.DataFrame(
            resid[0], index=self.data.index[:-1], columns=self.shocks)

    pars = np.array([s[0] for s in sample])

    edict = {'pars': pars.squeeze(),
             'means': means,
             'obs': obs,
             'covs': covs,
             'resid': resid,
             'flags': flags}

    return edict
