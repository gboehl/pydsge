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


def swarms(self, algos, linear=None, pop_size=100, maxgen=500, mig_share=.1, seed=None, tol_calls=None, use_ring=False, ncores=None, crit_mem=.85, update_freq=None, verbose=False, debug=False):

    import pygmo as pg
    import dill
    import pathos
    import random

    if crit_mem is not None:

        # TODO: one of the functions exposed by C++ leaks memory...
        import psutil
        if crit_mem < 1:
            crit_mem *= 100

    if linear is None:
        linear = self.linear_filter

    if seed is None:
        seed = self.fdict['seed']

    if tol_calls is None:
        tol_calls = maxgen

    if update_freq is None:
        update_freq = 0

    np.random.seed(seed)
    random.seed(seed)

    # globals are *evil*
    global lprob_global

    # import the global function and pretend it is defined on top level
    def lprob_local(par):
        return lprob_global(par, linear, verbose)

    sfunc_inst = GPP(lprob_local, self.fdict['prior_bounds'])

    class Swarm(object):

        name = 'Swarm'
        # NOTE: the swarm does NOT hold actual pygmo objects such as pygmo.population or pygmo.algorithm, but only the serialized versions thereof

        def __init__(self, algo, pop, seed=None):

            self.res = None
            self.ncalls = 0
            self.history = []

            self.pop = pop
            self.algo = algo
            self.seed = seed

            return

        def extract(self):

            if not debug:
                self.algo, self.pop = self.res.get()
            else:
                self.algo, self.pop = self.res

            return

        @property
        def ready(self):

            if self.res is None:
                return True
            else:
                return self.res.ready()

        @property
        def sname(self):

            algo = dill.loads(self.algo)
            aname = algo.get_name()
            sname = aname.split(':')[0]

            return sname + '_' + str(self.seed)

    def dump_pop(pop):

        xs = pop.get_x()
        fs = pop.get_f()
        sd = pop.get_seed()

        return (xs, fs, sd)

    def load_pop(ser_pop):

        xs, fs, sd = ser_pop
        pop_size = len(fs)

        prob = pg.problem(sfunc_inst)
        pop = pg.population(prob, size=pop_size, seed=sd)

        for i in range(pop_size):
            pop.set_xf(i, xs[i], fs[i])

        return pop

    def gen_pop(seed, algos, pop_size):

        random.seed(seed)
        algo = random.sample(algos, 1)[0]
        algo.set_seed(seed)
        prob = pg.problem(sfunc_inst)
        pop = pg.population(prob, size=pop_size, seed=seed)

        ser_pop = dump_pop(pop)
        ser_algo = dill.dumps(algo)

        return ser_algo, ser_pop

    def evolve(ser_algo, ser_pop):

        algo = dill.loads(ser_algo)
        pop = load_pop(ser_pop)

        pop = algo.evolve(pop)

        return dill.dumps(algo), dump_pop(pop),

    print('[swarms:]'.ljust(30, ' ') +
          ' Number of evaluations is %sx the generation length.' % (maxgen*pop_size))

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    if not debug:
        pool = pathos.pools.ProcessPool(ncores)
        pool.clear()

    mig_abs = int(pop_size*mig_share)

    print('[swarms:]'.ljust(30, ' ') +
          ' Creating overlord of %s swarms...' % ncores, end="", flush=True)

    if not debug:
        rests = [pool.apipe(gen_pop, s, algos, pop_size)
                            for s in range(ncores)]
        overlord = [Swarm(*res.get(), s)
                          for s, res in zip(range(ncores), rests)]
    else:
        rests = [gen_pop(s, algos, pop_size) for s in range(ncores)]
        overlord = [Swarm(*res, s) for s, res in zip(range(ncores), rests)]

    # better clear here already
    pool.clear()

    print('done.')
    print('[swarms:]'.ljust(30, ' ') + ' Swarming out! Bzzzzz...')

    done = False
    best_x = None

    xsw = np.empty((ncores, self.ndim))
    fsw = np.empty((ncores, 1))
    nsw = np.empty((ncores, 1), dtype=object)

    pbar = tqdm.tqdm(total=maxgen, dynamic_ncols=True)
    # pbar = tqdm.tqdm(total=ncalls*ncores, dynamic_ncols=True)

    ll_max = -np.inf

    while not done:
        for s in overlord:
            if not use_ring:
                if not debug and not s.ready:
                    continue

                # if s.ncalls >= ncalls:
                if pbar.n >= maxgen:
                    break

            if s.res is not None:
                # post-procssing
                s.extract()

                xs = s.pop[0]
                fs = s.pop[1]
                fas = fs[:, 0].argsort()

                # record history
                s.history.append(xs[fas][0])

                s.ncalls += 1
                pbar.update()

                # keep us informed
                ll_max_swarm = -fs[fas][0][0]
                if ll_max_swarm > ll_max:
                    ll_max = ll_max_swarm
                    ll_max_cnt = pbar.n

                pbar.set_description('ll: '+str(ll_max_swarm.round(5)).rjust(
                    12, ' ')+' ['+str(ll_max.round(5))+'/'+str(ll_max_cnt)+']')

                # keep us up to date
                if update_freq and pbar.n and not pbar.n % update_freq:

                    for i, sw in enumerate(overlord):
                        fsw[i, :] = -sw.pop[1].min()
                        xsw[i, :] = sw.pop[0][s.pop[1].argmin()]
                        nsw[i, :] = sw.sname

                    swarms = xsw, fsw, nsw.reshape(1, -1)
                    pbar.write(str(summary(
                        swarms, self['__data__']['estimation']['prior'], swarm_mode=True, show_priors=False)))

                if ll_max_cnt < pbar.n - ncores*tol_calls and ll_max == ll_max_swarm:
                    print('[swarms:]'.ljust(
                        30, ' ') + ' No improvement in the last %s calls, exiting...' % tol_calls)
                    done=True
                    break

                # migrate the worst
                # if best_x is not None and s.ncalls < ncalls:
                if best_x is not None and pbar.n < maxgen:
                    for no, x, f in zip(fas[-mig_abs:], best_x, best_f):
                        s.pop[0][no]=x
                        s.pop[1][no]=f

                # save best for the next
                best_x=xs[fas][:mig_abs]
                best_f=fs[fas][:mig_abs]

            # if s.ncalls < ncalls:
            if pbar.n < maxgen:

                if crit_mem is not None:
                    # check if mem usage is above threshold
                    if psutil.virtual_memory()[2] > crit_mem:

                        pool.close()
                        print('[swarms:]'.ljust(20, ' ') + " Critical memory usage of "+str(
                            crit_mem)+"% reached, closing pools for maintenance...", end = "", flush = True)
                        pool.join()
                        print('fixing...', end = "", flush = True)
                        pool.restart()
                        print('done.', end = "", flush = True)

                if not debug:
                    s.res=pool.apipe(evolve, s.algo, s.pop)
                else:
                    s.res=evolve(s.algo, s.pop)

        # done = done or all([s.ncalls >= ncalls for s in overlord])
        done=done or pbar.n >= maxgen

    pbar.close()

    hs=[]

    for i, s in enumerate(overlord):
        fsw[i, :]=-s.pop[1].min()
        xsw[i, :]=s.pop[0][s.pop[1].argmin()]
        nsw[i, :]=s.sname
        hs.append(np.array(s.history))

    self.overlord=overlord
    self.par_cand=xsw
    self.swarms=xsw, fsw, nsw.reshape(1, -1)
    self.swarm_history=hs

    self.fdict['swarms']=xsw, fsw, nsw.reshape(1, -1)
    self.fdict['swarm_history']=hs

    return
