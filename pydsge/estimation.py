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
from .mpile import get_par, set_par


def prep_estim(self, N=None, linear=None, load_R=False, seed=None, eval_priors=False, dispatch=False, ncores=None, l_max=3, k_max=16, verbose=True, debug=False, **filterargs):
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
    elif linear:
        l_max, k_max = 1, 0

    if seed is None:
        if 'seed' in self.fdict.keys():
            seed = self.fdict['seed']
        else:
            seed = 0

    if not 'reduced_form' in filterargs:
        filterargs['reduced_form'] = True

    self.fdict['filter_n'] = N
    self.fdict['linear'] = linear
    self.fdict['seed'] = seed

    self.debug |= debug
    # self.Z = np.array(self.data)

    set_par(self, 'prior_mean', verbose=verbose > 3, l_max=l_max, k_max=k_max)

    self.create_filter(
        N=N, ftype='KalmanFilter' if linear else None, **filterargs)

    if 'filter_R' in self.fdict.keys():
        self.filter.R = self.fdict['filter_R']
    elif load_R:
        raise AttributeError('[estimation:]'.ljust(
            15, ' ') + "`filter.R` not in `fdict`.")

    # dry run before the fun beginns
    if np.isinf(get_ll(self, verbose=verbose > 3, dispatch=dispatch)):
        raise ValueError('[estimation:]'.ljust(
            15, ' ') + 'likelihood of initial values is zero.')

    if verbose:
        print('[estimation:]'.ljust(15, ' ') + ' Model operational. %s states, %s observables, %s shocks, %s data points.' %
              (len(self.vv), len(self.observables), len(self.shocks), len(self.data)))

    prior = self.prior
    par_fix = self.par_fix.copy()
    prior_arg = self.prior_arg

    # add to class so that it can be stored later
    self.fdict['prior_names'] = [pp for pp in prior.keys()]

    self.ndim = len(prior.keys())

    if 'frozen_prior' not in self.fdict.keys() or eval_priors:

        pfrozen, pinitv, bounds = get_prior(prior, verbose=verbose)
        self.fdict['frozen_prior'] = pfrozen
        self.fdict['prior_bounds'] = bounds
        self.fdict['init_value'] = pinitv

        if verbose:
            print('[estimation:]'.ljust(
                15, ' ') + ' %s priors detected. Adding parameters to the prior distribution.' % self.ndim)

    def llike(parameters, par_fix, linear, verbose, seed):

        random_state = np.random.get_state()
        with warnings.catch_warnings(record=True):
            try:
                warnings.filterwarnings('error')

                np.random.seed(seed)

                par_fix[prior_arg] = parameters
                par_active_lst = list(par_fix)

                # the gen_sys and following part replicates call to set_par, redundant
                self.gen_sys(par=par_active_lst, l_max=l_max,
                             k_max=k_max, verbose=verbose > 3)
                self.filter.Q = self.QQ(self.ppar) @ self.QQ(self.ppar)

                ll = get_ll(self, verbose=verbose > 3, dispatch=dispatch)

                np.random.set_state(random_state)
                return ll

            except KeyboardInterrupt:
                raise

            except Exception as err:
                if verbose:
                    print('[llike:]'.ljust(15, ' ') +
                          ' Failure. Error msg: %s' % err)
                    if verbose > 1:
                        pardict = get_par(self, full=False, asdict=True)
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

    def lprob(par, par_fix=par_fix, linear=None, verbose=verbose > 1, temp=1, lprob_seed='set'):

        lp = lprior(par)

        if np.isinf(lp):
            if verbose:
                print('[lprob:]'.ljust(15, ' ') + " prior is -inf.")
            return lp

        if linear is None:
            linear = linear_pa

        if verbose:
            st = time.time()

        if lprob_seed in ('vec', 'rand'):
            seed_loc = sum(p // 10**(int(np.log(abs(p))/np.log(10))-9)
                           for p in par)
            if lprob_seed == 'rand':
                seed_loc += np.random.randint(2**31)  # win explodes with 2**32
            seed_loc = int(seed_loc) % (2**31)  # win explodes with 2**32

        elif lprob_seed == 'set':
            seed_loc = seed
        else:
            raise NotImplementedError(
                "`lprob_seed` must be one of `('vec', 'rand', 'set')`.")

        ll = llike(par, par_fix, linear, verbose, seed_loc)*temp if temp else 0

        if np.isinf(ll):
            return ll

        ll += lp

        if verbose:
            print('[lprob:]'.ljust(15, ' ') + " Sample took %ss, ll is %s, temp is %s." %
                  (np.round(time.time() - st, 3), np.round(ll, 4), np.round(temp, 4)))

        return ll

    # make functions accessible
    self.lprob = lprob
    self.lprior = lprior
    self.llike = llike

    if ncores is None or ncores:
        create_pool(self, ncores)
    else:
        self.pool = None

    return


def create_pool(self, ncores=None, threadpool_limit=None):
    """Creates a reusable pool

    Parameters
    ----------

    ncores : int, optional
        Number of cores. Defaults to pathos' default, which is the number of cores.
    threadpool_limit : int, optional
        Number of threads that numpy uses independently of pathos. Only used if `threadpoolctl` is installed. Defaults to one.
    """

    import pathos

    if hasattr(self, 'pool'):
        ncores = ncores or self.pool.ncpus
        self.pool.close()
    else:
        # pools should carry their count with them
        ncores = ncores or pathos.multiprocessing.cpu_count()

    if threadpool_limit:
        self.threadpool_limit = threadpool_limit
    elif not hasattr(self, 'threadpool_limit'):
        self.threadpool_limit = 1

    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=self.threadpool_limit)
    except ImportError:
        print('[create_pool:]'.ljust(
            15, ' ') + " Could not import package `threadpoolctl` to limit numpy multithreading. This might reduce multiprocessing performance.")

    self.pool = pathos.pools.ProcessPool(ncores)
    self.pool.clear()

    return self.pool


@property
def mapper(self):

    if hasattr(self, 'pool') and not self.debug:
        return self.pool.imap
    else:
        return map
