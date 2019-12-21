#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
from grgrlib import fast0, map2arr
from .engine import boehlgorithm
from decimal import Decimal


@property
def linear_representation(self):
    """Get a linear representation of the system under the current parameters
    """
    from .core import get_sys

    mat = self.precalc_mat[0]
    dim_x = self.sys[2].shape[0]

    return mat[1, 0, 1][dim_x:]


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
    return state @ self.hx[0].T + self.hx[1]


def get_eps(self, x, xp):
    return (x - self.t_func(xp)[0]) @ self.SIG


def irfs(self, shocklist, pars=None, T=30, linear=False, verbose=False):
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

    if isinstance(shocklist, tuple):
        shocklist = [shocklist,]

    st = time.time()

    def irfs_runner(par):

        if np.any(par):
            self.set_par(par, autocompile=False)

        st_vec = np.zeros(len(self.vv))

        X = np.empty((T, len(self.vv)))
        K = np.empty(T)
        L = np.empty(T)

        superflag = False

        for t in range(T):

            shk_vec = np.zeros(len(self.shocks))
            for vec in shocklist:
                if vec[2] == t:

                    shock = vec[0]
                    shocksize = vec[1]

                    shock_arg = self.shocks.index(shock)
                    shk_vec[shock_arg] = shocksize

            st_vec, (l, k), flag = self.t_func(
                st_vec, shk_vec, linear=linear, return_k=True)

            superflag |= flag

            X[t,:] = st_vec
            L[t] = l
            K[t] = k


        return X, K, L, superflag

    if np.any(pars):
        res = self.mapper(irfs_runner, pars)
        X, K, L, flag = map2arr(res)
    else:
        X, K, L, flag = irfs_runner(pars)
        X = pd.DataFrame(X, columns=self.vv) 

    if np.any(flag) and verbose:
        print('[irfs:]'.ljust(15, ' ') + 'No rational expectations solution found at least once.')

    if verbose:
        print('[irfs:]'.ljust(15, ' ') + 'Simulation took ', np.round((time.time() - st), 5), ' seconds.')

    return X, (K, L)


def simulate(self, eps=None, mask=None, state=None, linear=False, verbose=False):
    """Simulate time series given a series of exogenous innovations.

    Parameters
    ----------
        eps : array
            Shock innovations of shape (T, n_eps)>
        mask : array
            Mask for eps. Each non-None element will be replaced.
        state : array
            Inital state.
    """

    if eps is None:
        eps = self.res.copy()

    if mask is not None:
        eps = np.where(np.isnan(mask), eps, mask*eps)

    if state is None:
        try:
            state = self.means[0]
        except:
            state = np.zeros(len(self.vv))

    X = [state]
    K = [0]
    L = [0]
    superflag = False

    if verbose:
        st = time.time()

    for eps_t in eps:

        state, (l, k), flag = self.t_func(
            state, noise=eps_t, return_k=True, linear=linear)

        superflag |= flag

        X.append(state)
        K.append(k)
        L.append(l)

    X = np.array(X)
    K = np.array(K)
    L = np.array(L)

    self.simulated_X = X

    if verbose:
        print('[simulate:]'.ljust(15, ' ')+'Simulation took ',
              time.time() - st, ' seconds.')

    if superflag and verbose:
        print('[simulate:]'.ljust(
            15, ' ')+'No rational expectations solution found.')

    return X, np.expand_dims(K, 2), superflag


def simulate_ts(self, T=1e3, cov=None, verbose=False):

    import scipy.stats as ss

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
