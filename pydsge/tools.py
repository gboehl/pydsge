#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time
from grgrlib import fast0
from .engine import boehlgorithm
from decimal import Decimal


@property
def linear_representation(self):
    """Get a linear representation of the system under the current parameters
    """

    get_sys(self, l_max=1, k_max=0)
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
              ' Transition function took %.2Es.' % Decimal(time.time() - st))

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


def irfs(self, shocklist, wannasee=None, horizon=30, linear=False, verbose=False):

    # REWRITE!!
    # returns time series of impule responses
    # shocklist: takes list of tuples of (shock, size, timing)
    # wannasee: list of strings of the variables to be plotted and stored

    slabels = self.vv
    olabels = self.observables

    args_sts = []
    args_obs = []

    if wannasee not in (None, 'all', 'full'):
        for v_raw in wannasee:
            v = v_raw.replace('_', '')
            if v in slabels:
                args_sts.append(list(slabels).index(v))
            elif v in olabels:
                args_obs.append(olabels.index(v))
            else:
                raise Exception(
                    "Variable %s neither in states nor observables. You might want to call self.get_sys() with the 'reduce_sys = False' argument. Note that underscores '_' are discarged." % v)

    st_vec = np.zeros(len(self.vv))

    Y = []
    K = []
    L = []
    superflag = False

    st = time.time()

    for t in range(horizon):

        shk_vec = np.zeros(len(self.shocks))
        for vec in shocklist:
            if vec[2] == t:

                shock = vec[0]
                shocksize = vec[1]

                shock_arg = self.shocks.index(shock)
                shk_vec[shock_arg] = shocksize

                shk_process = (self.SIG @ shk_vec).nonzero()

                for shk in shk_process:
                    args_sts += list(shk)

        st_vec, (l, k), flag = self.t_func(
            st_vec, shk_vec, linear=linear, return_k=True)

        if flag:
            superflag = True

        Y.append(st_vec)
        K.append(k)
        L.append(l)

    if superflag and verbose:
        print('[irfs:]'.ljust(15, ' ') +
              ' No rational expectations solution found.')

    Y = np.array(Y)
    K = np.array(K)
    L = np.array(L)

    care_for_sts = np.unique(args_sts)
    care_for_obs = np.unique(args_obs)

    if wannasee is None:
        Z = (self.hx[0] @ Y.T).T + self.hx[1]
        tt = ~fast0(Z-Z.mean(axis=0), 0)
        llabels = list(np.array(self.observables)[
                       tt])+list(self.vv[care_for_sts])
        X2 = Y[:, care_for_sts]
        X = np.hstack((Z[:, tt], X2))
    elif wannasee is 'full':
        llabels = list(self.vv)
        X = Y
    elif wannasee is 'all':
        tt = ~fast0(Y-Y.mean(axis=0), 0)
        llabels = list(self.vv[tt])
        X = Y[:, tt]
    else:
        llabels = list(self.vv[care_for_sts])
        X = Y[:, care_for_sts]
        if care_for_obs.size:
            llabels = list(np.array(self.observables)[care_for_obs]) + llabels
            Z = ((self.hx[0] @ Y.T).T + self.hx[1])[:, care_for_obs]
            X = np.hstack((Z, X))

    if verbose:
        print('[irfs:]'.ljust(15, ' ')+'Simulation took ',
              np.round((time.time() - st), 5), ' seconds.')

    labels = []
    for l in llabels:
        if not isinstance(l, str):
            l = l.name
        labels.append(l)

    return X, labels, (Y, K, L)


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
            15, ' ')+' No rational expectations solution found.')

    return X, np.expand_dims(K, 2), superflag


def simulate_series(self, T=1e3, cov=None, verbose=False):

    import scipy.stats as ss

    if cov is None:
        cov = self.QQ(self.par)

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
                  ' No rational expectations solution found.')

    return np.array(states), np.array(Ks)
