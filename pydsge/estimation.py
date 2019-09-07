#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
import os
import time
import emcee
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
            print('[bayesian_estimation -> pmdm:]'.ljust(30, ' ') +
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
            print('[bayesian_estimation -> pmdm:]'.ljust(30, ' ') +
                  " Maximizing posterior mode density using '%s' (meanwhile warnings are disabled)." % self.method)
        else:
            print('[bayesian_estimation -> pmdm:]'.ljust(30, ' ') +
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
                print('[bayesian_estimation -> '+self.desc_str+'pmdm:]'.ljust(45, ' ')+str(res['message']) +
                      ' Maximization returned value lower than actual (known) optimum ('+str(-self.res_max)+' > '+str(-self.res)+').')
            else:
                print('[bayesian_estimation -> '+self.desc_str+'pmdm:]'.ljust(45, ' ')+str(res['message']
                                                                                           )+' Log-likelihood is '+str(np.round(-res['fun'], 5))+'.')
            print('')

        except StopIteration:
            if not self.verbose:
                self.pbar.close()
            print('')
            print('[bayesian_estimation -> '+self.desc_str+'pmdm:]'.ljust(45, ' ') +
                  ' Maximum number of function calls exceeded, exiting. Log-likelihood is '+str(np.round(-self.res_max, 5))+'...')
            print('')

        except KeyboardInterrupt:
            if not self.verbose:
                self.pbar.close()
            print('')
            print('[bayesian_estimation -> '+self.desc_str+'pmdm:]'.ljust(45, ' ') +
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


def prep_estim(self, N=300, linear=False, seed=0, obs_cov=None, init_with_pmeans=False, verbose=False):
    """Initializes the tools necessary for estimation

    ...

    Parameters
    ----------
    obs_cov : ndarray, optional
        obeservation covariance. Defaults to 0.1 of the standard deviation of the time series
        If a float is given, thi will govern the fraction of the standard deviation.
    """

    # all that should be reproducible
    np.random.seed(seed)

    self.fdict['seed'] = seed

    if hasattr(self, 'data'):
        self.fdict['data'] = self.data
    elif 'data' in self.fdict.keys():
        self.data = self.fdict['data']

    self.Z = np.array(self.data)

    if obs_cov is None:
        obs_cov = 1e-1

    if isinstance(obs_cov, float):
        obs_cov = self.create_obs_cov(obs_cov)

    if not hasattr(self, 'sys'):
        self.get_sys(reduce_sys=True, verbose=verbose)

    self.preprocess(verbose=verbose > 1)

    self.create_filter(N=N, linear=linear, random_seed=seed)

    self.filter.R = obs_cov
    self.fdict['obs_cov'] = obs_cov

    # dry run before the fun beginns
    if np.isinf(self.get_ll(verbose=verbose)):
        raise ValueError('[bayesian_estimation:]'.ljust(
            30, ' ') + ' likelihood of initial values is zero.')

    print()
    print('[bayesian_estimation:]'.ljust(30, ' ') +
          ' Model operational. %s states, %s observables.' % (len(self.vv), len(self.observables)))
    print()

    par_fix = np.array(self.par).copy()

    p_names = [p.name for p in self.parameters]
    priors = self['__data__']['estimation']['prior']
    prior_arg = [p_names.index(pp) for pp in priors.keys()]

    # add to class so that it can be stored later
    self.fdict['prior_names'] = [pp for pp in priors.keys()]
    self.priors = priors
    self.par_fix = par_fix
    self.prior_arg = prior_arg

    self.ndim = len(priors.keys())

    if 'frozen_priors' not in self.fdict.keys():

        pfrozen, pinitv, bounds = get_priors(priors)
        self.fdict['frozen_priors'] = pfrozen
        self.fdict['prior_bounds'] = bounds

        if init_with_pmeans:
            self.init_par = [priors[pp][1] for pp in priors.keys()]
        else:
            self.init_par = pinitv
            for i in range(self.ndim):
                if pinitv[i] is None:
                    self.init_par[i] = par_fix[prior_arg][i]

        self.par_cand = self.init_par.copy()
        self.fdict['init_par'] = self.init_par

    print('[bayesian_estimation:]'.ljust(30, ' ') +
          ' %s priors detected. Adding parameters to the prior distribution.' % self.ndim)

    def llike(parameters, linear, verbose):

        if verbose == 2:
            st = time.time()

        with warnings.catch_warnings(record=True):
            try:
                warnings.filterwarnings('error')

                np.random.seed(seed)

                par_fix[prior_arg] = parameters
                par_active_lst = list(par_fix)

                self.get_sys(par=par_active_lst, reduce_sys=True,
                             verbose=verbose > 1)

                if not linear:
                    if self.linear_filter:
                        self.create_filter(N=N, linear=False, random_seed=seed)
                    # these max vals should be sufficient given we're only dealing with stochastic linearization
                    self.preprocess(l_max=3, k_max=16, verbose=verbose > 1)
                    self.filter.fx = self.t_func
                    self.filter.hx = self.o_func
                else:
                    if not self.linear_filter:
                        self.create_filter(linear=True, random_seed=seed)
                    self.preprocess(l_max=1, k_max=0, verbose=False)
                    self.filter.F = self.linear_representation()
                    self.filter.H = self.hx

                ll = self.get_ll(verbose=verbose)

                if verbose == 2:
                    print('[bayesian_estimation -> llike:]'.ljust(45, ' ') +
                          ' Sample took '+str(np.round(time.time() - st, 3))+'s.')

                return ll

            except KeyboardInterrupt:
                raise

            except Exception as err:
                if verbose == 2:
                    print('[bayesian_estimation -> llike:]'.ljust(45, ' ') +
                          ' Sample took '+str(np.round(time.time() - st, 3))+'s. (failure, error msg: %s)' % err)

                return -np.inf

    def lprior(pars):

        prior = 0
        for i, pl in enumerate(self.fdict['frozen_priors']):
            prior += pl.logpdf(pars[i])

        return prior

    def lprob(pars, linear, verbose):
        return lprior(pars) + llike(pars, linear, verbose)

    global lprob_global
    lprob_global = lprob

    # also make functions accessible
    self.lprob = lprob
    self.lprior = lprior
    self.llike = llike


def pmdm(self, linear=None, maxfev=None, linear_pre_pmdm=False, method=None, tol=1e-2, update_freq=None, verbose=False):

    if maxfev is None:
        maxfev = 1000

    if linear is None:
        linear = self.linear_filter

    if linear_pre_pmdm:
        print('[bayesian_estimation -> pmdm:]'.ljust(45, ' ') +
              ' starting pre-maximization of linear function.')
        self.par_cand = PMDM(self, maxfev, tol, method,
                             True, update_freq, verbose=verbose).go()
        print('[bayesian_estimation -> pmdm:]'.ljust(45, ' ') +
              ' pre-maximization of linear function done, starting actual maximization.')

    description = self.description

    self.pmdm_par = PMDM(self, maxfev, tol, method, linear,
                         update_freq, verbose=verbose).go()

    self.fdict['pmdm_par'] = self.pmdm_par

    self.par_cand = self.pmdm_par.copy()

    np.warnings.filterwarnings('default')

    print()
    print('[bayesian_estimation:]'.ljust(30, ' ')+' posterior mode values:')
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


def swarms(self, algos, linear=None, pop_size=100, ncalls=10, mig_share=.1, seed=None, use_ring=False, ncores=None, verbose=False):

    import pygmo as pg
    import dill
    import pathos
    import random

    if linear is None:
        linear = self.linear_filter

    if seed is None:
        seed = self.fdict['seed']

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

            self.algo, self.pop = self.res.get()

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

    print('[swarm_find:]'.ljust(30, ' ') +
          ' Number of iterations per core is %sx the generation length.' % (ncalls*pop_size))

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    pool = pathos.pools.ProcessPool(ncores)

    mig_abs = int(pop_size*mig_share)

    print('[swarm_find:]'.ljust(30, ' ') +
          ' Creating overlord of %s swarms...' % ncores, end="", flush=True)

    rests = [pool.apipe(gen_pop, s, algos, pop_size) for s in range(ncores)]
    overlord = [Swarm(*res.get(), s) for s, res in zip(range(ncores), rests)]

    print('done.')
    print('[swarm_find:]'.ljust(30, ' ') + ' Swarming out! Bzzzzz...')

    done = False
    best_x = None

    pbar = tqdm.tqdm(total=ncalls*ncores, dynamic_ncols=True)

    ll_max = -np.inf

    while not done:
        for s in overlord:
            if not use_ring:
                if not s.ready:
                    continue

                if s.ncalls >= ncalls:
                    continue

            if s.res is not None:
                # post-procssing
                s.extract()

                xs = s.pop[0]
                fs = s.pop[1]
                fas = fs[:, 0].argsort()

                # record history
                s.history.append(xs[fas][0])

                s.ncalls += 1

                # keep us informed
                ll_max_swarm = -fs[fas][0][0]
                ll_max = max(ll_max, ll_max_swarm)

                pbar.update()
                pbar.set_description(
                    'll: '+str(ll_max_swarm.round(5)).rjust(12, ' ')+' ['+str(ll_max.round(5))+']')

                # migrate the worst
                if best_x is not None and s.ncalls < ncalls:
                    for no, x, f in zip(fas[-mig_abs:], best_x, best_f):
                        s.pop[0][no] = x
                        s.pop[1][no] = f

                # save best for the next
                best_x = xs[fas][:mig_abs]
                best_f = fs[fas][:mig_abs]

            if s.ncalls < ncalls:

                s.res = pool.apipe(evolve, s.algo, s.pop)

        done = all([s.ncalls >= ncalls for s in overlord])

    pbar.close()

    xs = np.empty((ncores, self.ndim))
    fs = np.empty((ncores, 1))
    ns = np.empty((ncores, 1), dtype=object)
    hs = np.empty((ncores, ncalls, self.ndim))

    for i, s in enumerate(overlord):
        fs[i, :] = -s.pop[1].min()
        xs[i, :] = s.pop[0][s.pop[1].argmin()]
        ns[i, :] = s.sname
        hs[i, :] = np.array(s.history)

    self.overlord = overlord
    self.par_cand = xs
    self.swarms = xs, fs, ns.reshape(1, -1)
    self.swarm_history = hs

    self.fdict['swarms'] = xs, fs, ns.reshape(1, -1)
    self.fdict['swarm_history'] = hs

    return


def mcmc(self, nsteps=3000, nwalks=None, tune=None, seed=None, ncores=None, backend_file=None, linear=None, distr_init_chains=False, resume=False, update_freq=None, verbose=False, debug=False):

    import pathos

    if seed is None:
        seed = self.fdict['seed']

    if tune is None:
        # self.tune = int(nsteps*4/5.)
        ## 2/3 seems to be a better fit, given that we initialize at a good approximation of the posterior distribution
        self.tune = int(nsteps*2/3.)

    if update_freq is None:
        update_freq = int(nsteps/10.)

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    if linear is None:
        linear = self.linear_filter

    if backend_file is None:
        if 'backend_file' in self.fdict.keys():
            self.backend_file = str(self.fdict['backend_file'])
        elif hasattr(self, 'path') and hasattr(self, 'name'):
            self.backend_file = self.path + self.name + '_sampler.h5'
        else:
            print('Sampler will not be recorded.')
    else:
        self.backend_file = backend_file

    backend = emcee.backends.HDFBackend(self.backend_file)

    if not resume:
        backend.reset(nwalks, self.ndim)

    if nwalks is None:
        if resume:
            nwalks = backend.get_chain().shape[1]
        else:
            nwalks = 100

    if 'description' in self.fdict.keys():
        self.description = self.fdict['description']

    # globals are *evil*
    global lprob_global

    # import the global function and pretend it is defined on top level
    def lprob_local(par):
        return lprob_global(par, linear, verbose)

    loc_pool = pathos.pools.ProcessPool(ncores)
    loc_pool.clear()

    if debug:
        sampler = emcee.EnsembleSampler(nwalks, self.ndim, lprob_local)
    else:
        sampler = emcee.EnsembleSampler(
            nwalks, self.ndim, lprob_local, pool=loc_pool, backend=backend)

    if resume:
        p0 = sampler.get_last_sample()
    elif distr_init_chains:

        print()
        print('[bayesian_estimation:]'.ljust(30, ' ') +
              ' finding initial values for mcmc (distributed over priors):')
        p0 = np.empty((nwalks, self.ndim))
        pbar = tqdm.tqdm(total=nwalks, unit='init.val(s)', dynamic_ncols=true)

        for w in range(nwalks):
            draw_prob = -np.inf

            while np.isinf(draw_prob):
                nprr = np.random.randint
                # alternatively on could use enumerate() on frozen_priors and include the itarator in the random_state
                pdraw = [pl.rvs(random_state=nprr(2**32-1))
                         for pl in self.fdict['frozen_priors']]
                draw_prob = lprob(pdraw, linear, verbose)

            p0[w, :] = np.array(pdraw)
            pbar.update(1)

        pbar.close()

    else:
        if np.ndim(self.par_cand) > 1:

            p0 = np.empty((nwalks, self.ndim))
            cand_dim = self.par_cand.shape[0]

            for i, w in enumerate(range(nwalks)):
                par = self.par_cand[i % cand_dim]
                p0[w, :] = par * (1+1e-3*np.random.randn())

        else:
            p0 = self.par_cand*(1+1e-3*np.random.randn(nwalks, self.ndim))

    if not verbose:
        np.warnings.filterwarnings('ignore')

    if not verbose:
        pbar = tqdm.tqdm(total=nsteps, unit='sample(s)',
                         dynamic_ncols=True)
        report = pbar.write
    else:
        report = print

    for result in sampler.sample(p0, iterations=nsteps):

        cnt = sampler.iteration

        if cnt and update_freq and not cnt % update_freq:

            report('')
            if self.description is not None:
                report('[bayesian_estimation -> mcmc:]'.ljust(45, ' ') +
                       ' Summary from last %s of %s iterations (%s):' % (update_freq, cnt, str(self.description)))

            else:
                report('[bayesian_estimation -> mcmc:]'.ljust(45, ' ') +
                       ' Summary from last %s of %s iterations:' % (update_freq, cnt))

            sample = sampler.get_chain()
            report(str(summary(sample, self.priors, tune=-update_freq).round(3)))
            report("Mean likelihood is %s, mean acceptance fraction is %s." % (lprob_local(
                np.mean(sample, axis=(0, 1))).round(3), np.mean(sampler.acceptance_fraction).round(2)))

        if not verbose:
            pbar.update(1)

    pbar.close()

    if not verbose:
        np.warnings.filterwarnings('default')

    print("mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)))

    self.sampler = sampler

    return
