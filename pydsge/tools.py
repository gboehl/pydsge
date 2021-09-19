#!/bin/python
# -*- coding: utf-8 -*-

"""module for model simulations
"""

import os
import numpy as np
import pandas as pd
import time
import tqdm
from grgrlib import fast0, map2arr
from grgrlib.multiprocessing import serializer
from .engine import *
from decimal import Decimal


def t_func(self, state, shocks=None, set_k=None, return_flag=None, return_k=False, get_obs=False, linear=False, verbose=False):
    """transition function

    Parameters
    ----------
    state : array 
        full state in y-space
    shocks : array, optional
        shock vector. If None, zero vector will be assumed (default)
    set_k : tuple of int, optional
        set the expected number of periods if desired. Otherwise will be calculated endogenoulsy (default).
    return_flag : bool, optional
        wheter to return error flags, defaults to True
    return_k : bool, optional
        wheter to return values of (l,k), defaults to False
    linear : bool, optional
        wheter to ignore the constraint and return the linear solution, defaults to False
    verbose : bool or int, optional
        Level of verbosity, defaults to 0
    """

    if verbose:
        st = time.time()

    omg, lam, x_bar = self.sys
    pmat, qmat, pterm, qterm, bmat, bterm = self.precalc_mat

    dimp, dimq = omg.shape
    dimeps = self.neps

    if shocks is None:
        shocks = np.zeros(dimeps)

    if linear:
        set_k = 0
        set_l = 0
    elif set_k is None or isinstance(set_k, bool):
        set_k = -1
        set_l = -1
    elif isinstance(set_k, tuple):
        set_l, set_k = (int(set_v) for set_v in set_k)
    else:
        set_l = int(not bool(set_k))
        set_k = int(set_k)

    if return_flag is None:
        return_flag = True

    pobs, q, l, k, flag = t_func_jit(pmat, pterm, qmat[:, :, :-dimeps], qterm[..., :-dimeps],
                                     bmat, bterm, x_bar, *self.hx, state[-dimq+dimeps:], shocks, set_l, set_k, get_obs)

    newstate = (q, pobs) if get_obs else np.hstack((pobs, q))

    if verbose:
        print('[t_func:]'.ljust(15, ' ') +
              'Transition function took %.2Es.' % Decimal(time.time() - st))

    if return_k:
        return newstate, (l, k), flag
    elif return_flag:
        return newstate, flag
    else:
        return newstate


def o_func(self, state, covs=None, pars=None):
    """Get observables from state representation

    Parameters
    ----------
    state : array
    covs : array, optional
        Series of covariance matrices. If provided, 95% intervals will be calculated, including the intervals of the states
    """

    if pars is not None:

        obs = []
        for sti, par in zip(state, pars):
            self.set_par(par, get_hx_only=True)
            ob = sti[:, :self.dimp] @ self.hx[0].T + \
                sti[:, self.dimp:] @ self.hx[1].T + self.hx[2]
            obs.append(ob)

        return np.array(obs)

    try:
        obs = state[..., :self.dimp] @ self.hx[0].T + \
            state[..., self.dimp:] @ self.hx[1].T + self.hx[2]
    except ValueError as e:
        raise ValueError(
            str(e) + ' you probably want to use the filter/`load_estim` with `reduced_form=False`.')

    if np.ndim(state) <= 1:
        data = self.data.index if hasattr(self, 'data') else None
        obs = pd.DataFrame(obs, index=data, columns=self.observables)

    if covs is None:
        return obs

    var = np.diagonal(covs, axis1=1, axis2=2)
    std = np.sqrt(var)
    iv95 = np.stack((state - 1.96*std, state, state + 1.96*std))

    std_obs = (np.hstack((self.hx[0], self.hx[1])) @ std.T).T
    iv95_obs = np.stack((obs - 1.96*std_obs, obs, obs + 1.96*std_obs))

    return iv95_obs, iv95


def irfs(self, shocklist, pars=None, state=None, T=30, linear=False, set_k=False, force_init_equil=None, verbose=True, debug=False, **args):
    """Simulate impulse responses

    Parameters
    ----------

    shocklist : tuple or list of tuples
        Tuple of (shockname, size, period)
    T : int, optional
        Simulation horizon. (default: 30)
    linear : bool, optional
        Simulate linear model (default: False)
    set_k: int, optional
        Enforce a `k` (defaults to False)
    force_init_equil:
        If set to `False`, the equilibrium will be recalculated every iteration. This may be problematic if there is multiplicity because the algoritm selects the equilibrium with the lowest (l,k) (defaults to True)
    verbose : bool or int, optional
        Level of verbosity (default: 1)

    Returns
    -------
    DataFrame, tuple(int,int)
        The simulated series as a pandas.DataFrame object and the expected durations at the constraint
    """

    self.debug |= debug
    if force_init_equil is None:
        force_init_equil = not bool(np.any(set_k))

    if not isinstance(shocklist, list):
        shocklist = [shocklist, ]

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    st = time.time()
    shocks = self.shocks
    nstates = self.dimx

    if self.set_par is not None:
        set_par = serializer(self.set_par)
    t_func = serializer(self.t_func)

    # accept all sorts of inputs
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

        supererrflag = False
        supermultflag = False
        l, k = 0, 0

        for t in range(T):

            shk_vec = np.zeros(len(shocks))
            for vec in new_shocklist:
                if vec[2] == t:

                    shock = vec[0]
                    shocksize = vec[1]

                    shock_arg = shocks.index(shock)
                    shk_vec[shock_arg] = shocksize

            # force_init_equil will force recalculation of l,k only if the shock vec is not empty
            if force_init_equil and not np.any(shk_vec):
                set_k_eff = (l-1, k) if l else (l, max(k-1, 0))

                _, (l_endo, k_endo), flag = t_func(
                    st_vec[-(self.dimq-self.dimeps):], shk_vec, set_k=None, linear=linear, return_k=True)

                multflag = l_endo != set_k_eff[0] or k_endo != set_k_eff[1]
                supermultflag |= multflag

                if verbose > 1 and multflag:
                    print('[irfs:]'.ljust(
                        15, ' ') + 'Multiplicity found in period %s: new eql. %s coexits with old eql. %s.' % (t, (l_endo, k_endo), set_k_eff))

            elif set_k is None:
                set_k_eff = None
            elif isinstance(set_k, tuple):
                set_l_eff, set_k_eff = set_k
                if set_l_eff-t >= 0:
                    set_k_eff = set_l_eff-t, set_k_eff
                else:
                    set_k_eff = 0, max(set_k_eff+set_l_eff-t, 0)
            elif set_k:
                set_k_eff = 0, max(set_k-t, 0)
            else:
                set_k_eff = set_k

            if set_k_eff:
                if set_k_eff[0] > self.lks[0] or set_k_eff[1] > self.lks[1]:
                    raise IndexError(
                        'set_k exceeds l_max (%s vs. %s).' % (set_k_eff, self.lks))

            st_vec, (l, k), flag = t_func(
                st_vec[-(self.dimq-self.dimeps):], shk_vec, set_k=set_k_eff, linear=linear, return_k=True)

            if flag and verbose > 1:
                print('[irfs:]'.ljust(
                    15, ' ') + 'No OBC solution found in period %s (error flag %s).' % (t, flag))

            supererrflag |= flag

            X[t, :] = st_vec
            L[t] = l
            K[t] = k

        return X, L, K, supererrflag, supermultflag

    if pars is not None and np.ndim(pars) > 1:
        res = self.mapper(runner, pars)
        X, L, K, flag, multflag = map2arr(res)
    else:
        X, L, K, flag, multflag = runner(pars)
        X = pd.DataFrame(X, columns=self.vv)

    if verbose == 1:
        if np.any(flag):
            print('[irfs:]'.ljust(14, ' ') +
                  ' No OBC solution(s) found.')
        elif np.any(multflag):
            print('[irfs:]'.ljust(14, ' ') +
                  ' Multiplicity/Multiplicities found.')

    if verbose > 2:
        print('[irfs:]'.ljust(15, ' ') + 'Simulation took ',
              np.round((time.time() - st), 5), ' seconds.')

    return X, np.vstack((L, K)), flag


def shock2state(self, shock):
    """create state vector given shock and size
    """

    stype, ssize = shock[:2]
    state = np.zeros(self.dimq)
    state[-self.dimeps:][list(self.shocks).index(stype)] = ssize

    return state


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


def simulate(self, source=None, mask=None, pars=None, resid=None, init=None, operation=np.multiply, linear=False, debug=False, verbose=False, **args):
    """Simulate time series given a series of exogenous innovations.

    Parameters
    ----------
        source : dict
            Dict of `extract` results
        mask : array
            Mask for eps. Each non-None element will be replaced.
    """
    pars = pars if pars is not None else source['pars']
    resi = resid if resid is not None else source['resid']
    init = init if init is not None else source['init']

    sample = pars, resi, init

    if verbose:
        st = time.time()

    self.debug |= debug

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    if self.set_par is not None:
        set_par = serializer(self.set_par)
    else:
        set_par = None

    t_func = serializer(self.t_func)
    obs = serializer(self.obs)
    vv_orig = self.vv.copy()

    def runner(arg):

        superflag = False
        par, eps, state = arg

        if mask is not None:
            eps = np.where(np.isnan(mask), eps, operation(np.array(mask), eps))

        if set_par is not None:
            _, vv = set_par(par, return_vv=True, **args)
            if not np.all(vv == vv_orig):
                raise Exception(
                    'The ordering of variables has changed given different parameters.')

        X = [state]
        L, K = [], []

        for eps_t in eps:

            state, (l, k), flag = t_func(
                state, eps_t, return_k=True, linear=linear)

            superflag |= flag

            X.append(state)
            L.append(l)
            K.append(k)

        X = np.array(X)
        LK = np.array((L, K))
        K = np.array(K)

        return X, LK, superflag

    wrap = tqdm.tqdm if verbose else (lambda x, **kwarg: x)
    res = wrap(self.mapper(runner, zip(*sample)), unit=' sample(s)',
               total=len(source['pars']), dynamic_ncols=True)

    X, LK, flags = map2arr(res)

    if verbose > 1:
        print('[simulate:]'.ljust(15, ' ')+'Simulation took ',
              time.time() - st, ' seconds.')

    if np.any(flags) and verbose:
        print('[simulate:]'.ljust(
            15, ' ')+'No OBC solution found (at least once).')

    return X, (LK[..., 0, :], LK[..., 1, :]), flags


def traj(self, state, l=None, k=None, verbose=True):

    if k is None or l is None:

        omg, lam, x_bar = self.sys
        pmat, qmat, pterm, qterm, bmat, bterm = self.precalc_mat
        l, k, flag = find_lk(bmat, bterm, x_bar, state)

        if verbose:
            if flag == 0:
                meaning = ''
            elif flag == 1:
                meaning = ' (no solution)'
            elif flag == 2:
                meaning = ' (no solution, k_max reached)'

            print('[traj:]'.ljust(15, ' ') +
                  'l=%s, k=%s, flag is %s%s.' % (l, k, flag, meaning))

    if not hasattr(self, 'precalc_tmat'):

        fq1, fp1, fq0 = self.ff
        preprocess_tmats(self, fq1, fp1, fq0, verbose > 1)

    tmat, tterm = self.precalc_tmat

    return tmat[:, l, k-1] @ state + tterm[:, l, k-1]


def k_map(self, state, l=None, k=None, verbose=True):

    omg, lam, x_bar = self.sys

    if k is None:
        pmat, qmat, pterm, qterm, bmat, bterm = self.precalc_mat
        l_endo, k, flag = find_lk(bmat, bterm, x_bar, state)

        l = l or l_endo

        if verbose:
            if flag == 0:
                meaning = ''
            elif flag == 1:
                meaning = ' (no solution)'
            elif flag == 2:
                meaning = ' (no solution, k_max reached)'

            print('[k_map:]'.ljust(15, ' ') +
                  'l=%s, k=%s, flag is %s%s.' % (l, k, flag, meaning))
    else:
        l = l or 0

    if not hasattr(self, 'precalc_tmat'):

        fq1, fp1, fq0 = self.ff
        preprocess_tmats(self, fq1, fp1, fq0, verbose > 1)

    l_max, k_max = self.lks
    tmat, tterm = self.precalc_tmat

    LS = np.array([tmat[i+k, i, k] @ state + tterm[i+k, i, k]
                   for i in range(l_max)])
    KS = np.array([tmat[l+i, l, i] @ state + tterm[l+i, l, i]
                   for i in range(k_max)])
    return LS - x_bar, KS - x_bar
