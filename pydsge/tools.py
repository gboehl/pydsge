#!/bin/python
# -*- coding: utf-8 -*-

"""contains functions related to simulating the mode
"""

import os
import numpy as np
import pandas as pd
import time
from grgrlib import fast0, map2arr
from .engine import boehlgorithm
from decimal import Decimal


@property
def lin_t_func(self):
    """Get a linear representation of the system under the current parameters
    """
    mat = self.precalc_mat[0]
    dim_x = self.sys[2].shape[0]

    return mat[1, 0, 1][dim_x:]


@property
def lin_o_func(self):
    """Get a linear representation of the observation function under the current parameters
    """
    return self.hx


def t_func(self, state, noise=None, return_flag=True, return_k=False, linear=False, verbose=False):

    if verbose:
        st = time.time()

    newstate = state.copy()

    if noise is not None:
        newstate += self.SIG @ noise
    newstate, (l, k), flag = boehlgorithm(self, newstate, linear=linear)

    if verbose:
        print('[t_func:]'.ljust(15, ' ') +
              'Transition function took %.2Es.' % Decimal(time.time() - st))

    if return_k:
        return newstate, (l, k), flag
    elif return_flag:
        return newstate, flag
    else:
        return newstate


def o_func(self, state):
    """
    observation function
    """

    obs = state @ self.hx[0].T + self.hx[1]
    if np.ndim(state) <= 1:
        data = self.data.index if hasattr(self, 'data') else None
        obs = pd.DataFrame(obs, index=data, columns=self.observables)

    return obs


def calc_obs(self, states, covs=None):
    """Get observables from state representation

    Parameters
    ----------
    states : array
    covs : array, optional
        Series of covariance matrices. If provided, 95% intervals will be calculated.
    """

    if covs is None:
        return states @ self.hx[0].T + self.hx[1]

    var = np.diagonal(covs, axis1=1, axis2=2)
    std = np.sqrt(var)
    iv95 = np.stack((states - 1.96*std, states, states + 1.96*std))

    obs = (self.hx[0] @ states.T).T + self.hx[1]
    std_obs = (self.hx[0] @ std.T).T
    iv95_obs = np.stack((obs - 1.96*std_obs, obs, obs + 1.96*std_obs))

    return iv95_obs, iv95


def irfs(self, shocklist, pars=None, state=None, T=30, linear=False, verbose=True, **args):
    """Simulate impulse responses

    Parameters
    ----------

    shocklist : tuple or list of tuples
        Tuple of (shockname, size, period)
    T : int
        Simulation horizon. (default: 30)

    Returns
    -------
    DataFrame, tuple(int,int)
        The simulated series as a pandas.DataFrame object and the expected durations at the constraint
    """

    from grgrlib.core import serializer

    if isinstance(shocklist, tuple):
        shocklist = [shocklist, ]

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    st = time.time()
    set_par = serializer(self.set_par)
    t_func = serializer(self.t_func)
    shocks = self.shocks
    nstates = len(self.vv)

    def runner(par):

        X = np.empty((T, nstates))
        K = np.empty(T)
        L = np.empty(T)

        if np.any(par):
            try:
                set_par(par, **args)
            except ValueError:
                X[:] = np.nan
                K[:] = np.nan
                L[:] = np.nan
                return X, K, L, 4

        st_vec = state if state is not None else np.zeros(nstates)

        superflag = False

        for t in range(T):

            shk_vec = np.zeros(len(shocks))
            for vec in shocklist:
                if vec[2] == t:

                    shock = vec[0]
                    shocksize = vec[1]

                    shock_arg = shocks.index(shock)
                    shk_vec[shock_arg] = shocksize

            st_vec, (l, k), flag = t_func(
                st_vec, shk_vec, linear=linear, return_k=True)

            superflag |= flag

            X[t, :] = st_vec
            L[t] = l
            K[t] = k

        return X, K, L, superflag

    if pars is not None and np.ndim(pars) > 1:
        res = self.mapper(runner, pars)
        X, K, L, flag = map2arr(res)
    else:
        X, K, L, flag = runner(pars)
        X = pd.DataFrame(X, columns=self.vv)

    if np.any(flag) and verbose:
        print('[irfs:]'.ljust(15, ' ') +
              'No rational expectations solution found at least once.')

    if verbose > 1:
        print('[irfs:]'.ljust(15, ' ') + 'Simulation took ',
              np.round((time.time() - st), 5), ' seconds.')

    return X, (K, L), flag


@property
def mask(self, verbose=False):

    if verbose:
        print('[mask:]'.ljust(15, ' ') + 'Shocks:', self.shocks)

    msk = self.data.copy()
    msk[:] = np.nan

    try:
        self.observables
    except AttributeError:
        raise AttributeError("Model not initialized. Try calling `set_par` first. Cheers.")

    return msk.rename(columns=dict(zip(self.observables, self.shocks)))[:-1]


def simulate(self, source, mask=None, pars=None, resid=None, init=None, linear=False, debug=False, verbose=False, **args):
    """Simulate time series given a series of exogenous innovations.

    Parameters
    ----------
        source : dict
            Dict of `extract` results
        mask : array
            Mask for eps. Each non-None element will be replaced.
    """
    from grgrlib.core import serializer

    pars = pars or source['pars']
    resi = resid or source['resid']
    init = init or [s[0] for s in source['means']]

    sample = zip(pars, resi, init)

    if verbose:
        st = time.time()

    self.debug |= debug

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    set_par = serializer(self.set_par)
    t_func = serializer(self.t_func)
    obs = serializer(self.obs)

    def runner(arg):

        superflag = False
        par, eps, state = arg

        if mask is not None:
            eps = np.where(np.isnan(mask), eps, np.array(mask)*eps)

        set_par(par, **args)

        X = [state]
        Y = [obs(state)]
        K = []
        L = []

        for eps_t in eps:

            state, (l, k), flag = t_func(
                state, noise=eps_t, return_k=True, linear=linear)

            superflag |= flag

            X.append(state)
            Y.append(obs(state))
            L.append(l)
            K.append(k)

        X = np.array(X)
        Y = np.array(Y)
        L = np.array(L)
        K = np.array(K)

        return X, Y, (L, K), superflag

    wrap = tqdm.tqdm if verbose else (lambda x, **kwarg: x)

    res = wrap(self.mapper(runner, sample), unit=' sample(s)',
               total=len(source['pars']), dynamic_ncols=True)

    res = map2arr(res)
    superflag = res[-1].any()

    if verbose:
        print('[simulate:]'.ljust(15, ' ')+'Simulation took ',
              time.time() - st, ' seconds.')

    if superflag and verbose:
        print('[simulate:]'.ljust(
            15, ' ')+'No rational expectations solution found.')

    X, Y, LK, flags = res

    return X, Y, (LK[:, 0, :], LK[:, 1, :]), flags


def simulate_ts(self, par=None, cov=None, T=1e3, verbose=False, **args):
    """Simulate a random time series (probably not up-to-date)
    """

    import scipy.stats as ss

    if par is None:
        self.set_par(par, **args)

    if cov is None:
        cov = self.QQ(self.ppar)

    st_vec = np.zeros(len(self.vv))

    states, Ks = [], []
    for i in range(int(T)):
        shk_vec = ss.multivariate_normal.rvs(cov=cov)
        print(shk_vec)
        st_vec, ks, flag = self.t_func(
            st_vec, shk_vec, return_k=True, verbose=verbose)
        states.append(st_vec)
        Ks.append(ks)

        if verbose and flag:
            print('[irfs:]'.ljust(15, ' ') +
                  'No rational expectations solution found.')

    return np.array(states), np.array(Ks)
