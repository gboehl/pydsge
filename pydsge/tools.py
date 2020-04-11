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


def t_func(self, state, noise=None, set_k=None, return_flag=True, return_k=False, linear=False, verbose=False):

    if verbose:
        st = time.time()

    newstate = state.copy()

    if noise is not None:
        newstate += self.SIG @ noise

    if set_k is None or isinstance(set_k,bool):
        newstate, (l, k), flag = boehlgorithm(self, newstate, linear=linear)

    else:
        print('woah')
        mat, term, _, _ = self.precalc_mat
        J = self.sys[2]
        l,k = int(not bool(set_k)), set_k
        flag = 0

        newstate = (mat[l, k, 1] @ newstate + term[l, k, 1])[J.shape[0]:]

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


def irfs(self, shocklist, pars=None, state=None, T=30, linear=False, set_k=False, verbose=True, debug=False, **args):
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

    self.debug |= debug

    if not isinstance(shocklist, list):
        shocklist = [shocklist, ]

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    st = time.time()
    shocks = self.shocks
    nstates = len(self.vv)

    set_par = serializer(self.set_par)
    t_func = serializer(self.t_func)

    ## accept all sorts of inputs
    new_shocklist = []

    for vec in shocklist:
        if isinstance(vec, str):
            vec = (vec, 1, 0) 
        elif len(vec) == 2:
            vec += 0,
        new_shocklist.append(vec)

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
            for vec in new_shocklist:
                if vec[2] == t:

                    shock = vec[0]
                    shocksize = vec[1]

                    shock_arg = shocks.index(shock)
                    shk_vec[shock_arg] = shocksize

            set_k_eff = max(set_k-t,0) if set_k else set_k

            st_vec, (l, k), flag = t_func(st_vec, shk_vec, set_k=set_k_eff, linear=linear, return_k=True)

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
        raise AttributeError(
            "Model not initialized. Try calling `set_par` first. Cheers.")

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
    init = init or np.array(source['means'])[...,0,:]

    sample = pars, resi, init

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
        LK = np.array((L,K))
        K = np.array(K)

        return X, Y, LK, superflag

    wrap = tqdm.tqdm if verbose else (lambda x, **kwarg: x)

    if np.ndim(resi) > 2 or np.ndim(pars) > 1 or np.ndim(init) > 2:

        res = wrap(self.mapper(runner, zip(*sample)), unit=' sample(s)', total=len(source['pars']), dynamic_ncols=True)
        res = map2arr(res)

    else:
        res = runner(sample)

    superflag = np.any(res[-1])

    if verbose:
        print('[simulate:]'.ljust(15, ' ')+'Simulation took ',
              time.time() - st, ' seconds.')

    if superflag and verbose:
        print('[simulate:]'.ljust(
            15, ' ')+'No rational expectations solution found.')

    X, Y, LK, flags = res

    return X, Y, (LK[..., 0, :], LK[..., 1, :]), flags


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
        st_vec, ks, flag = self.t_func(
            st_vec, shk_vec, return_k=True, verbose=verbose)
        states.append(st_vec)
        Ks.append(ks)

        if verbose and flag:
            print('[irfs:]'.ljust(15, ' ') +
                  'No rational expectations solution found.')

    return np.array(states), np.array(Ks)
