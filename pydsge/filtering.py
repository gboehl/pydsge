#!/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import scipy.linalg as sl
import pandas as pd
from econsieve import KalmanFilter, TEnKF
from grgrlib.core import timeprint
from grgrlib.multiprocessing import serializer
from econsieve.stats import logpdf


def create_obs_cov(self, scale_obs=0.1):

    self.Z = np.array(self.data)
    sig_obs = np.var(self.Z, axis=0)*scale_obs**2
    obs_cov = np.diagflat(sig_obs)

    return obs_cov


def get_p_init_lyapunov(self, Q):

    pmat = self.precalc_mat[0]
    qmat = self.precalc_mat[1]

    F = np.vstack((pmat[1, 0][:, :-self.neps],
                   qmat[1, 0][:-self.neps, :-self.neps]))

    E = np.vstack((pmat[1, 0][:, -self.neps:],
                  qmat[1, 0][:-self.neps, -self.neps:]))
    Q = E @ Q @ E.T

    p4 = sl.solve_discrete_lyapunov(F[self.dimp:,:], Q[self.dimp:,self.dimp:])

    return F @ p4 @ F.T + Q


def get_eps_lin(self, x, xp, rcond=1e-14):
    """Get filter-implied (smoothed) shocks for linear model
    """

    qmat = self.precalc_mat[1]

    if self.filter.name == 'KalmanFilter':
        pmat = self.precalc_mat[0]
        F = self.filter.F
        E = np.vstack((pmat[1, 0][:, -self.neps:],
                       qmat[1, 0][:-self.neps, -self.neps:]))

    else:
        F = qmat[1, 0][:, :-self.neps]
        E = qmat[1, 0][:, -self.neps:]

    return np.linalg.pinv(E, rcond) @ (x - F@xp)


def create_filter(self, R=None, N=None, ftype=None, seed=None, incl_obs=False, reduced_form=False, **fargs):

    self.Z = np.array(self.data)

    if ftype == 'KalmanFilter':
        ftype = 'KF'
    elif ftype == 'ParticleFilter':
        ftype = 'PF'
    elif ftype == 'AuxiliaryParticleFilter':
        ftype = 'APF'

    if ftype == 'KF':

        f = KalmanFilter(dim_x=self.dimx, dim_z=self.nobs)

    elif ftype in ('PF', 'APF'):

        print(
            'Warning: Particle filter is experimental and currently not under development.')
        from .pfilter import ParticleFilter

        if N is None:
            N = 10000

        aux_bs = ftype == 'APF'
        f = ParticleFilter(N=N, dim_x=self.dimx,
                           dim_z=self.nobs, auxiliary_bootstrap=aux_bs)

    else:
        ftype = 'TEnKF'

        if N is None:
            N = int((self.dimq-self.dimeps)**2/2) + 1

        dimx = self.dimq-self.dimeps if reduced_form else self.dimx
        f = TEnKF(N=N, dim_x=dimx, dim_z=self.nobs, seed=seed, **fargs)
        f.reduced_form = reduced_form

    if R is not None:
        f.R = R

    # use lyapunov equation as default. Otherwise to be defined manually via `*.filter.p`
    f.init_cov = None

    try:
        f.Q = self.QQ(self.ppar) @ self.QQ(self.ppar)
    except AttributeError:
        f.Q = self.fdict['QQ'] @ self.fdict['QQ']
    self.filter = f

    return f


def get_ll(self, **args):
    return run_filter(self, smoother=False, get_ll=True, **args)


def run_filter(self, smoother=True, get_ll=False, init_cov=None, dispatch=None, rcond=1e-14, seed=None, verbose=False):

    if verbose:
        st = time.time()

    self.Z = np.array(self.data)
    dimp = self.dimp

    if init_cov is not None:
        self.filter.init_cov = init_cov 

    # assign current transition & observation functions (of parameters)
    if self.filter.name == 'KalmanFilter':

        pmat = self.precalc_mat[0]
        qmat = self.precalc_mat[1]

        F = np.vstack((pmat[1, 0][:, :-self.neps],
                       qmat[1, 0][:-self.neps, :-self.neps]))
        F = np.pad(F, ((0, 0), (dimp, 0)))

        self.filter.F = F
        self.filter.H = np.hstack((self.hx[0], self.hx[1])), self.hx[2]

        if self.filter.Q.shape[0] == self.neps:
            E = np.vstack((pmat[1, 0][:, -self.neps:],
                          qmat[1, 0][:-self.neps, -self.neps:]))
            self.filter.Q = E @ self.filter.Q @ E.T

        if self.filter.init_cov is None:
            p4 = sl.solve_discrete_lyapunov(F[dimp:,dimp:], self.filter.Q[dimp:,dimp:])
            self.filter.P = F[:,dimp:] @ p4 @ F.T[dimp:] + self.filter.Q
        else:
            self.filter.P = self.filter.init_cov 

    elif dispatch or self.filter.name == 'ParticleFilter':
        from .engine import func_dispatch
        t_func_jit, o_func_jit, get_eps_jit = func_dispatch(self, full=True)
        self.filter.t_func = t_func_jit
        self.filter.o_func = o_func_jit
        self.filter.get_eps = get_eps_jit

    elif self.filter.reduced_form:
        self.filter.t_func = lambda *x: self.t_func(*x, get_obs=True)
        self.filter.o_func = None

        if self.filter.init_cov is None:
            qmat = self.precalc_mat[1]
            F = qmat[1, 0][:-self.neps, :-self.neps]
            E = qmat[1, 0][:-self.neps, -self.neps:]

            Q = E @ self.filter.Q @ E.T

            self.filter.P = sl.solve_discrete_lyapunov(F, Q)
        else:
            self.filter.P = self.filter.init_cov 

    else:
        self.filter.t_func = self.t_func
        self.filter.o_func = self.o_func

        if self.filter.init_cov is None:
            self.filter.P = get_p_init_lyapunov(self, self.filter.Q)
        else:
            self.filter.P = self.filter.init_cov

    self.filter.get_eps = self.get_eps_lin

    if self.filter.name == 'KalmanFilter':

        means, covs, ll = self.filter.batch_filter(self.Z)

        if smoother:
            means, covs, _, _ = self.filter.rts_smoother(
                means, covs, inv=lambda x: np.linalg.pinv(x, rcond=rcond))

        if get_ll:
            res = ll
        else:
            means = means
            res = (means, covs)

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
            self.Z, calc_ll=get_ll, store=smoother, seed=seed, verbose=verbose > 0)

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


def extract(self, sample=None, nsamples=1, init_cov=None, precalc=True, seed=0, nattemps=4, accept_failure=False, verbose=True, debug=False, l_max=None, k_max=None, **npasargs):
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
    from grgrlib.core import map2arr

    if sample is None:
        if type(self).__name__ == "DSGE_DUMMY":
            sample = None
        else:
            sample = self.par

    if np.ndim(sample) <= 1:
        sample = [sample]

    np.random.seed(seed)

    fname = self.filter.name
    verbose = max(verbose, debug)

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    if fname == 'ParticleFilter':
        raise NotImplementedError

    elif fname == 'KalmanFilter':
        if nsamples > 1:
            print('[extract:]'.ljust(
                15, ' ')+' Setting `nsamples` to 1 as the linear filter does not rely on sampling.')
        nsamples = 1
        debug = not hasattr(self, 'debug') or self.debug
        self.debug = True

    else:
        if self.filter.reduced_form:
            self.create_filter(
                R=self.filter.R, N=self.filter.N, reduced_form=False)

            print('[extract:]'.ljust(
                15, ' ')+' Extraction requires filter in non-reduced form. Recreating filter instance.')

        npas = serializer(self.filter.npas)

    self.debug |= debug

    if sample[0] is not None:
        set_par = serializer(self.set_par)

    run_filter = serializer(self.run_filter)
    t_func = serializer(self.t_func)
    edim = len(self.shocks)
    xdim = len(self.vv)
    odim = len(self.observables)

    obs_func = serializer(self.obs)
    filter_get_eps = serializer(self.get_eps_lin)

    dimeps = self.dimeps
    dimp = self.dimp

    seeds = np.random.randint(2**31, size=nsamples)  # win explodes with 2**32
    sample = [(x, y) for x in sample for y in seeds]

    def runner(arg):

        par, seed_loc = arg

        if par is not None:
            set_par(par, l_max=l_max, k_max=k_max)

        res = run_filter(verbose=verbose > 2, seed=seed_loc, init_cov=init_cov)

        if fname == 'KalmanFilter':
            means, covs = res
            res = means.copy()
            resid = np.empty((means.shape[0]-1, dimeps))

            for t, x in enumerate(means[1:]):
                resid[t] = filter_get_eps(x, res[t])
                res[t+1] = t_func(res[t], resid[t], linear=True)[0]

            return par, res[0], resid, 0

        np.random.shuffle(res)
        sample = np.dstack((obs_func(res), res[..., dimp:]))
        inits = res[:, 0, :]

        def t_func_loc(states, eps):

            (q, pobs), flag = t_func(states, eps, get_obs=True)

            return np.hstack((pobs, q)), flag

        for natt in range(nattemps):
            try:
                init, resid, flags = npas(func=t_func_loc, X=sample, init_states=inits, verbose=max(
                    len(sample) == 1, verbose-1), seed=seed_loc, nsamples=1, **npasargs)

                return par, init, resid[0], flags

            except Exception as e:
                raised_error = e

        if accept_failure:
            print('[extract:]'.ljust(
                15, ' ') + "got an error: '%s' (after %s unsuccessful attemps)." % (raised_error, natt+1))
            return None
        else:
            import sys
            raise type(raised_error)(str(raised_error) + ' (after %s unsuccessful attemps).' %
                                     (natt+1)).with_traceback(sys.exc_info()[2])

    wrap = tqdm.tqdm if (verbose and len(sample) >
                         1) else (lambda x, **kwarg: x)
    res = wrap(self.mapper(runner, sample), unit=' sample(s)',
               total=len(sample), dynamic_ncols=True)
    pars, init, resid, flags = map2arr(res)

    if hasattr(self, 'pool') and self.pool:
        self.pool.close()

    if fname == 'KalmanFilter':
        self.debug = debug

    if resid.shape[0] == 1:
        resid[0] = pd.DataFrame(
            resid[0], index=self.data.index[:-1], columns=self.shocks)

    edict = {'pars': pars,
             'init': init,
             'resid': resid,
             'flags': flags}

    return edict
