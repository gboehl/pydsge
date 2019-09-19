#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
import os
import time
from .stats import get_priors, mc_mean, summary, pmdm_report
import scipy.optimize as so
import tqdm


class PMDM(object):
    """A wrapper to have a progress par for the posterior mode maximization.
    """

    name = 'PMDM'

    def __init__(self, model, maxfev, tol, method, linear, update_freq, verbose):

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
            self.x = self.model.par_cand

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

        return self.x_max


class GPP:
    """Generic PYGMO problem
    """

    name = 'GPP'

    def __init__(self, func, bounds):

        self.func = func
        self.bounds = bounds

    def fitness(self, x):
        return [-self.func(x)]

    def get_bounds(self):
        return self.bounds


def pmdm(self, linear=None, maxfev=None, linear_pre_pmdm=False, method=None, tol=1e-2, update_freq=None, verbose=False):

    if maxfev is None:
        maxfev = 1000

    if linear is None:
        linear = self.linear_filter

    if linear_pre_pmdm:
        print('[pmdm:]'.ljust(30, ' ') +
              ' starting pre-maximization of linear function.')
        self.par_cand = PMDM(self, maxfev, tol, method,
                             True, update_freq, verbose=verbose).go()
        print('[pmdm:]'.ljust(30, ' ') +
              ' pre-maximization of linear function done, starting actual maximization.')

    description = self.description

    self.pmdm_par = PMDM(self, maxfev, tol, method, linear,
                         update_freq, verbose=verbose).go()

    self.fdict['pmdm_par'] = self.pmdm_par

    self.par_cand = self.pmdm_par.copy()

    np.warnings.filterwarnings('default')

    print()
    print('[estimation:]'.ljust(30, ' ')+' posterior mode values:')
    with os.popen('stty size', 'r') as rows_cols:
        cols = rows_cols.read().split()[1]
        lnum = (len(self.priors)*8)//(int(cols)-8) + 1
        priors_chunks = np.array_split(
            np.array(self.fdict['prior_names']), lnum)
        vals_chunks = np.array_split([round(m_val, 3)
                                      for m_val in self.pmdm_par], lnum)
        for pchunk, vchunk in zip(priors_chunks, vals_chunks):
            row_format = "{:>8}" * (len(pchunk) + 1)
            print(row_format.format("", *pchunk))
            print(row_format.format("", *vchunk))
            print()

    print()

    return self.pmdm_par


def nlopt(self, p0=None, linear=None, maxfev=None, method=None, tol=1e-2, update_freq=None, verbose=False):

    from pydsge.estimation import GPP, get_init_par
    import pygmo as pg

    if linear is None:
        linear = self.linear_filter

    lprob = lambda x: self.lprob(x, linear=linear, verbose=verbose)

    sfunc_inst = GPP(lprob, self.fdict['prior_bounds'])

    if p0 is None:
        p0 = get_init_par(self, 1, linear, 0, False, verbose)

    if method is None:
        method = 'cobyla'

    algo = pg.algorithm(pg.nlopt(solver=method))

    if update_freq is None:
        update_freq = 1

    if update_freq:
        algo.set_verbosity(update_freq)

    if maxfev is not None:
        algo.extract(pg.nlopt).maxeval = maxfev

    prob = pg.problem(sfunc_inst)
    pop = pg.population(prob, 1)
    pop.set_x(0, np.squeeze(p0))
    pop = algo.evolve(pop)

    self.pmdm_par = pop.champion_x
    self.par_cand = pop.champion_x
    self.fdict['nlopt_champ'] = pop.champion_x

    return  pop.champion_x

