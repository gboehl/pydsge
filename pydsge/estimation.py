#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import pathos
import time
import tqdm
from .stats import get_prior
from .filtering import get_ll
from .core import get_par


def prep_estim(self, N=None, linear=None, load_R=False, seed=None, eval_priors=False, dispatch=False, constr_data=False, ncores=None, verbose=True, **filterargs):
    """Initializes the tools necessary for estimation

    ...

    Parameters
    ----------
    N : int, optional
        Number of ensemble members for the TEnKF. Defaults to 300 if no previous information is available.
    linear : bool, optional
        Whether a liniar or nonlinear filter is used. Defaults to False if no previous information is available.
    load_R : bool, optional
        Whether to load `filter.R` from prevous information. 
    seed : bool, optional
        Random seed. Defaults to 0
    dispatch : bool, optional
        Whether to use a dispatcher to create jitted transition and observation functions. Defaults to False.
    constr_data : bool, optional
        Whether to apply the constraint to the data as well. Defaults to False.
    verbose : bool/int, optional
        Whether display messages:
            0 - no messages
            1 - duration
            2 - duration & error messages
            3 - duration, error messages & vectors
            4 - maximum informative
    """

    import warnings

    # all that should be reproducible
    np.random.seed(seed)

    if N is None:
        if 'filter_n' in self.fdict.keys():
            N = int(self.fdict['filter_n'])
        else:
            N = 300

    if linear is None:
        if 'linear' in self.fdict.keys():
            linear = self.fdict['linear']
        else:
            linear = False

    if seed is None:
        if 'seed' in self.fdict.keys():
            seed = self.fdict['seed']
        else:
            seed = 0

    self.fdict['filter_n'] = N
    self.fdict['linear'] = linear
    self.fdict['seed'] = seed
    self.fdict['constr_data'] = constr_data

    self.Z = np.array(self.data)

    if not hasattr(self, 'sys') or not hasattr(self, 'precalc_mat'):
        self.get_sys(reduce_sys=True, verbose=verbose > 3)

    self.create_filter(
        N=N, ftype='KalmanFilter' if linear else None, **filterargs)

    if 'filter_R' in self.fdict.keys():
        self.filter.R = self.fdict['filter_R']
    elif load_R:
        raise AttributeError('[estimation:]'.ljust(
            15, ' ') + "`filter.R` not in `fdict`.")

    # dry run before the fun beginns
    if np.isinf(get_ll(self,constr_data=constr_data, verbose=verbose > 3, dispatch=dispatch)):
        raise ValueError('[estimation:]'.ljust(
            15, ' ') + 'likelihood of initial values is zero.')

    if verbose:
        print('[estimation:]'.ljust(15, ' ') + 'Model operational. %s states, %s observables.' %
              (len(self.vv), len(self.observables)))

    prior = self.prior
    par_fix = self.par_fix.copy()
    prior_arg = self.prior_arg

    # add to class so that it can be stored later
    self.fdict['prior_names'] = [pp for pp in prior.keys()]

    self.ndim = len(prior.keys())

    if 'frozen_prior' not in self.fdict.keys() or eval_priors:

        pfrozen, pinitv, bounds = get_prior(prior)
        self.fdict['frozen_prior'] = pfrozen
        self.fdict['prior_bounds'] = bounds
        self.fdict['init_value'] = pinitv

    if verbose:
        print('[estimation:]'.ljust(
            15, ' ') + '%s priors detected. Adding parameters to the prior distribution.' % self.ndim)

    def llike(parameters, linear, verbose, seed):

        random_state = np.random.get_state()
        with warnings.catch_warnings(record=True):
            try:
                warnings.filterwarnings('error')

                np.random.seed(seed)

                par_fix[prior_arg] = parameters
                par_active_lst = list(par_fix)

                if not linear:
                    if self.filter.name == 'KalmanFilter':
                        raise AttributeError('[estimation:]'.ljust(
                            15, ' ') + 'Missmatch between linearity choice (filter vs. lprob)')
                    # these max vals should be sufficient given we're dealing with stochastic linearization
                    self.get_sys(par=par_active_lst, l_max=3, k_max=16,
                                 reduce_sys=True, verbose=verbose > 3)
                    self.filter.Q = self.QQ(self.par) @ self.QQ(self.par)
                else:
                    if not self.filter.name == 'KalmanFilter':
                        raise AttributeError('[estimation:]'.ljust(
                            15, ' ') + 'Missmatch between linearity choice (filter vs. lprob)')
                    self.get_sys(par=par_active_lst, l_max=1, k_max=0,
                                 reduce_sys=True, verbose=verbose > 3)
                    self.filter.F = self.linear_representation
                    self.filter.H = self.hx

                    CO = self.SIG @ self.QQ(self.par)
                    self.filter.Q = CO @ CO.T

                ll = get_ll(self,constr_data=constr_data,
                                 verbose=verbose > 3, dispatch=dispatch)

                np.random.set_state(random_state)
                return ll

            except KeyboardInterrupt:
                raise

            except Exception as err:
                if verbose:
                    print('[llike:]'.ljust(15, ' ') +
                          'Failure. Error msg: %s' % err)
                    if verbose > 1:
                        pardict = get_par(self, full=False)
                        print(pardict)
                        self.box_check([*pardict.values()])

                np.random.set_state(random_state)
                return -np.inf

    def lprior(par):

        prior = 0
        for i, pl in enumerate(self.fdict['frozen_prior']):
            prior += pl.logpdf(par[i])

        return prior

    linear_pa = linear

    def lprob(par, linear=None, verbose=verbose, temp=1, lprob_seed='set'):

        lp = lprior(par)

        if np.isinf(lp):
            return lp

        if linear is None:
            linear = linear_pa

        if verbose:
            st = time.time()

        if lprob_seed in ('vec', 'rand'):
            seed_loc = sum(p // 10**(int(np.log(abs(p))/np.log(10))-9)
                           for p in par)
            if lprob_seed == 'rand':
                seed_loc += np.random.randint(2**32-2)
            seed_loc = int(seed_loc) % (2**32 - 1)

        elif lprob_seed == 'set':
            seed_loc = seed
        else:
            raise NotImplementedError(
                "`lprob_seed` must be one of `('vec', 'rand', 'set')`.")

        ll = llike(par, linear, verbose, seed_loc)*temp if temp else 0

        if np.isinf(ll):
            return ll

        ll += lp

        if verbose:
            print('[lprob:]'.ljust(15, ' ') + "Sample took %ss, ll is %s, temp is %s." %
                  (np.round(time.time() - st, 3), np.round(ll, 4), np.round(temp, 3)))

        return ll

    # make functions accessible
    self.lprob = lprob
    self.lprior = lprior
    self.llike = llike

    if ncores is None or ncores:
        self.pool = pathos.pools.ProcessPool(ncores)
        self.pool.clear()
    else:
        self.pool = None
