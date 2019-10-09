#!/bin/python
# -*- coding: utf-8 -*-

import particles
from particles import state_space_models as ssm
import numpy as np
from numba import njit, prange


HALFLOG2PI = 0.5 * np.log(np.pi)


@njit(parallel=True)
def logpdf_jit(x, get_eps, locs, L, logdet, dim):

    res = np.empty(locs.shape[0])

    for i in prange(locs.shape[0]):
        z = np.linalg.solve(L, get_eps(x, locs[i]).T)
        res[i] = - 0.5 * np.sum(z**2) - logdet - dim * HALFLOG2PI

    return res


@njit(parallel=True)
def rvs_jit(state, func, L, size, dim, dim_v):

    res = np.empty((size, dim_v))

    for i in prange(size):
        z = np.empty(dim)
        for j in range(dim):
            z[j] = np.random.normal()
        res[i] = func(state[i], np.dot(z, L.T))[0]

    return res


class StochTFunc(particles.distributions.ProbDist):
    def __init__(self, t_func, get_eps, state, cov):

        self.t_func = t_func
        self.get_eps = get_eps
        self.state = state
        self.cov = cov

        cov_error = ValueError('mvnorm: argument cov must be a dxd ndarray, \
                               with d>1, defining a symetric positive matrix')
        try:
            self.L = np.linalg.cholesky(cov)
            self.halflogdetcor = np.sum(np.log(np.diag(self.L)))
        except:
            raise cov_error
        if self.dim < 2 or cov.shape != (self.dim, self.dim):
            raise cov_error

    @property
    def dim(self):
        return self.cov.shape[0]

    @property
    def nstates(self):
        return self.state.shape[1]

    def logpdf(self, x):
        return logpdf_jit(x, self.get_eps, self.state, self.L, self.halflogdetcor, self.dim)

    def rvs(self, size=1):
        return rvs_jit(self.state, self.t_func, self.L, size, self.dim, self.nstates)


class DSGESSM(ssm.StateSpaceModel):

    def __init__(self, t_func, obs_func, get_eps, init_cov, t_cov, obs_cov):

        self.t_func = t_func
        self.obs_func = obs_func
        self.get_eps = get_eps

        self.init_cov = init_cov
        self.t_cov = t_cov
        self.obs_cov = obs_cov

    def PX0(self):
        # Distribution of X_0
        return particles.distributions.MvNormal(cov=self.init_cov)

    def PX(self, t, xp):
        # Distribution of X_t given X_{t-1}=xp (p=past)
        return StochTFunc(t_func=self.t_func, get_eps=self.get_eps, state=xp, cov=self.t_cov)

    def PY(self, t, xp, x):
        # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return particles.distributions.MvNormal(loc=self.obs_func(x), cov=self.obs_cov)


class ParticleFilter(object):

    name = 'ParticleFilter'

    def __init__(self, N, dim_x=None, dim_z=None, seed=None, auxiliary_bootstrap=True):

        self.N = N
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.seed = seed

        self.R = np.eye(self._dim_z)
        self.Q = np.eye(self._dim_z)
        self.P = np.eye(self._dim_x)

        self.data = None

        def dummy(x): return print('implement me')

        self.t_func = dummy
        self.o_func = dummy
        self.get_eps = dummy

        self.auxiliary_bootstrap = auxiliary_bootstrap

    @property
    def ss_mod(self):
        return DSGESSM(self.t_func, self.o_func, self.get_eps, self.P, self.Q, self.R)

    @property
    def fk_mod(self):
        if self.auxiliary_bootstrap:
            return ssm.AuxiliaryBootstrap(ssm=self.ss_mod, data=self.data)
        else:
            # AuxiliaryBootstrap seems to perform better
            return ssm.Bootstrap(ssm=self.ss_mod, data=self.data)

    def simulate(self, niter):

        true_states, sim_data = self.ss_mod.simulate(niter)
        true_states = np.array(true_states).reshape(niter, self._dim_x)
        sim_data = np.array(sim_data).squeeze()

        return true_states, sim_data

    def batch_filter(self, data=None):

        if data is not None:
            self.data = data

        self.pf = particles.SMC(
            fk=self.fk_mod, N=self.N, seed=self.seed, moments=False, store_history=True)
        self.pf.run()

        self.X = np.array(self.pf.hist.X).swapaxes(
            0, 1).reshape(self.N, len(self.data), self._dim_x)

        return self.X

    def smoother(self, nback=False):

        if nback > 1:
            smooth_trajectories = self.pf.hist.backward_sampling(nback)
            self.S = np.array(smooth_trajectories).swapaxes(0, 1).reshape(nback, len(self.data), self._dim_x)

        else:
            smooth_trajectories = self.pf.hist.extract_one_trajectory()
            self.S = np.array(smooth_trajectories)

        return self.S
