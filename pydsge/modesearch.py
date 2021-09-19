#!/bin/python
# -*- coding: utf-8 -*-

import os
import time
import tqdm
import numpy as np
from grgrlib.multiprocessing import serializer
from .mpile import get_par
from .stats import summary, pmdm_report


class PMDM(object):
    """A wrapper to have a progress par for the posterior mode maximization.
    """

    name = 'PMDM'

    def __init__(self, model, maxfev, tol, method, linear, update_freq, verbose):

        import scipy.optimize as so

        print('[pmdm:]'.ljust(15, ' ') + "WARNING: I have not used this function for quite a while, it is unmaintained and probably malfunctioning! `cmaes` is likely to do a better job.")

        self.model = model
        self.maxfev = maxfev
        self.tol = tol
        self.linear = linear
        self.update_freq = update_freq
        if update_freq is None:
            self.update_freq = int(maxfev*.1)
        self.verbose = verbose

        self.n = 0
        self.res_max = np.inf

        if not verbose:
            self.pbar = tqdm.tqdm(total=maxfev, dynamic_ncols=True)
            self.report = self.pbar.write
        else:
            self.report = print

        if linear:
            self.desc_str = 'linear_'
        else:
            self.desc_str = ''

        print()
        self.opt_dict = {}
        if method is None:
            self.method = 'Nelder-Mead'
        elif isinstance(method, int):
            methodl = ["Nelder-Mead", "Powell", "BFGS", "CG",
                       "L-BFGS-G", "SLSQP", "trust-constr", "COBYLA", "TNC"]

            # Nelder-Mead: fast and reliable, but doesn't max out the likelihood completely (not that fast if far away from max)
            # Powell: provides the highes likelihood but is slow and sometimes ends up in strange corners of the parameter space (sorting effects)
            # BFGS: hit and go but *can* outperform Nelder-Mead without sorting effects
            # CG: *can* perform well but can also get lost in a bad region with low LL
            # L-BFGS-G: leaves values untouched
            # SLSQP: fast but not very precise (or just wrong)
            # trust-constr: very fast but terminates too early
            # COBYLA: very fast but hangs up for no good reason and is effectively unusable
            # TNC: gets stuck around the initial values

            self.method = methodl[method]
            print('[pmdm:]'.ljust(20, ' ') +
                  ' Available methods are %s.' % ', '.join(methodl))
        if self.method == 'trust-constr':
            self.opt_dict = {'maxiter': np.inf}
        if self.method == 'Nelder-Mead':
            self.opt_dict = {
                'maxiter': np.inf,
                'maxfev': np.inf
            }
        if not verbose:
            np.warnings.filterwarnings('ignore')
            print('[pmdm:]'.ljust(20, ' ') +
                  " Maximizing posterior mode density using '%s' (meanwhile warnings are disabled)." % self.method)
        else:
            print('[pmdm:]'.ljust(20, ' ') +
                  ' Maximizing posterior mode density using %s.' % self.method)
        print()

    def __call__(self, pars):

        self.res = -self.model.lprob(pars, self.linear, self.verbose)
        self.x = pars

        # better ensure we're not just running with the wolfs when maxfev is hit
        if self.res < self.res_max:
            self.res_max = self.res
            self.x_max = self.x

        self.n += 1

        if not self.verbose:

            # ensure displayed number is correct
            self.pbar.n = self.n
            self.pbar.update(0)

            self.pbar.set_description(
                'll: '+str(-self.res.round(5)).rjust(12, ' ')+' ['+str(-self.res_max.round(5))+']')

        # prints information snapshots
        if self.update_freq and not self.n % self.update_freq:

            pmdm_report(self.model, self.x_max,
                        self.res_max, self.n, self.report)

        if self.n >= self.maxfev:
            raise StopIteration

        return self.res

    def go(self):

        try:
            f_val = -np.inf

            self.x = get_par('best', self, linear=linear,
                             verbose=verbose, full=False)

            res = so.minimize(self, self.x, method=self.method,
                              tol=self.tol, options=self.opt_dict)

            if not self.verbose:
                self.pbar.close()
            print('')
            if self.res_max < res['fun']:
                print('[pmdm ('+self.desc_str+'):]'.ljust(20, ' ')+str(res['message']) +
                      ' Maximization returned value lower than actual (known) optimum ('+str(-self.res_max)+' > '+str(-self.res)+').')
            else:
                print('[pmdm ('+self.desc_str+'):]'.ljust(20, ' ')+str(res['message']
                                                                       )+' Log-likelihood is '+str(np.round(-res['fun'], 5))+'.')
            print('')

        except StopIteration:
            if not self.verbose:
                self.pbar.close()
            print('')
            print('[pmdm ('+self.desc_str+'):]'.ljust(20, ' ') +
                  ' Maximum number of function calls exceeded, exiting. Log-likelihood is '+str(np.round(-self.res_max, 5))+'...')
            print('')

        except KeyboardInterrupt:
            if not self.verbose:
                self.pbar.close()
            print('')
            print('[pmdm ('+self.desc_str+'):]'.ljust(20, ' ') +
                  ' Iteration interrupted manually. Log-likelihood is '+str(np.round(-self.res_max, 5))+'...')
            print('')

        return self.x_max, self.res_max


def pmdm(self, linear=None, maxfev=None, linear_pre_pmdm=False, method=None, tol=1e-2, update_freq=None, verbose=False):

    print('[pmdm:]'.ljust(15, ' ') + "WARNING: I have not used this function for quite a while, it is unmaintained and probably malfunctioning! `cmaes` is likely to do a better job.")

    if maxfev is None:
        maxfev = 1000

    if linear is None:
        linear = self.linear_filter

    if linear_pre_pmdm:
        print('[pmdm:]'.ljust(30, ' ') +
              ' starting pre-maximization of linear function.')
        self.fdict['mode_x'] = PMDM(self, maxfev, tol, method,
                                    True, update_freq, verbose=verbose).go()
        print('[pmdm:]'.ljust(30, ' ') +
              ' pre-maximization of linear function done, starting actual maximization.')

    description = self.description

    self.pmdm_par, fmax = PMDM(self, maxfev, tol, method, linear,
                               update_freq, verbose=verbose).go()

    self.fdict['pmdm_x'] = self.pmdm_par
    self.fdict['pmdm_f'] = fmax

    if 'mode_f' in self.fdict.keys() and fmax < self.fdict['mode_f']:
        print('[pmdm:]'.ljust(15, ' ') + " New mode of %s is below old mode of %s. Rejecting..." %
              (fmax, self.fdict['mode_f']))
    else:
        self.fdict['mode_x'] = self.pmdm_par
        self.fdict['mode_f'] = fmax

    np.warnings.filterwarnings('default')

    print()
    print('[estimation:]'.ljust(30, ' ')+' posterior mode values:')
    with os.popen('stty size', 'r') as rows_cols:
        cols = rows_cols.read().split()[1]
        lnum = (len(self.prior)*8)//(int(cols)-8) + 1
        prior_chunks = np.array_split(
            np.array(self.fdict['prior_names']), lnum)
        vals_chunks = np.array_split([round(m_val, 3)
                                      for m_val in self.pmdm_par], lnum)
        for pchunk, vchunk in zip(prior_chunks, vals_chunks):
            row_format = "{:>8}" * (len(pchunk) + 1)
            print(row_format.format("", *pchunk))
            print(row_format.format("", *vchunk))
            print()

    print()

    return self.pmdm_par


def cmaes(self, p0=None, sigma=None, pop_size=None, restart_factor=2, seeds=3, seed=None, linear=None, lprob_seed=None, update_freq=1000, verbose=True, debug=False, **args):
    """Find mode using CMA-ES from grgrlib.

    Parameters
    ----------
    pop_size : int
        Size of each population. (Default: number of dimensions)
    seeds : in, optional
        Number of different seeds tried. (Default: 3)
    """

    from grgrlib.optimize import cmaes as fmin

    np.random.seed(seed or self.fdict['seed'])

    if isinstance(seeds, int):
        seeds = np.random.randint(2**32-2, size=seeds)

    bnd = np.array(self.fdict['prior_bounds'])
    p0 = get_par(self, 'adj_prior_mean', full=False,
                 asdict=False) if p0 is None else p0
    p0 = (p0 - bnd[0])/(bnd[1] - bnd[0])

    sigma = sigma or .25

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    self.debug |= debug
    lprob_global = serializer(self.lprob)

    def lprob(par): return lprob_global(
        par, linear=linear, lprob_seed=lprob_seed or 'set')

    def lprob_scaled(x): return -lprob((bnd[1] - bnd[0])*x + bnd[0])

    if self.pool:
        self.pool.clear()

    f_max = -np.inf

    print('[cma-es:]'.ljust(15, ' ') + 'Starting mode search over %s seeds...' %
          (seeds if isinstance(seeds, int) else len(seeds)))

    try:
        hist = self.fdict['cmaes_history']
        f_hist, x_hist = list(hist[0]), list(hist[1])
    except KeyError:
        f_hist = []
        x_hist = []

    for s in seeds:

        verbose = np.ceil(
            update_freq/pop_size) if update_freq is not None and pop_size is not None else None

        np.random.seed(s)
        res = fmin(lprob_scaled, p0, sigma, popsize=pop_size,
                   verbose=verbose, mapper=self.mapper, debug=debug, **args)

        x_scaled = res[0] * (bnd[1] - bnd[0]) + bnd[0]
        f_hist.append(-res[1])
        x_hist.append(x_scaled)

        if -res[1] < f_max:
            print('[cma-es:]'.ljust(15, ' ') +
                  'Current solution of %s rejected at seed %s.' % (np.round(-res[1], 4), s))

        else:
            f_max = -res[1]
            x_max_scaled = x_scaled
            # reinject solution
            p0 = res[0]
            if verbose:
                print('[cma-es:]'.ljust(15, ' ') +
                      'Updating best solution to %s at seed %s.' % (np.round(f_max, 4), s))

        # apply restart_factor
        if pop_size:
            pop_size *= restart_factor

        if verbose:
            from .clsmethods import mode_summary

            if self.description is not None:
                print('[cma-es:]'.ljust(15, ' ') +
                      'Searching %s (%s)' % (self.name, self.description))
            mode_summary(self, data_cmaes=(f_hist, x_hist))
            print('')

    np.warnings.filterwarnings('default')

    self.fdict['cmaes_mode_x'] = x_max_scaled
    self.fdict['cmaes_mode_f'] = f_max
    self.fdict['cmaes_history'] = f_hist, x_hist, seeds

    if 'mode_f' in self.fdict.keys() and f_max < self.fdict['mode_f']:
        print('[cmaes:]'.ljust(15, ' ') + " New mode of %s is below old mode of %s. Rejecting..." %
              (f_max, self.fdict['mode_f']))
    else:
        self.fdict['mode_x'] = x_max_scaled
        self.fdict['mode_f'] = f_max

    self.pool.close()

    return f_max, x_max_scaled
