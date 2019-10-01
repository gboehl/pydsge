#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
import os
import time
from .stats import get_priors, mc_mean, summary, pmdm_report
import scipy.optimize as so
import tqdm


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


def prep_estim(self, N=None, linear=None, seed=None, obs_cov=None, constr_data=False, init_with_pmeans=False, verbose=False):
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

    if N is None:
        if 'filter_n' in self.fdict.keys():
            N = int(self.fdict['filter_n'])
        else:
            N = 300

    if linear is None:
        if 'filter_lin' in self.fdict.keys():
            linear = self.fdict['filter_lin']
        else:
            linear = False


    if seed is None:
        if 'seed' in self.fdict.keys():
            seed = self.fdict['seed']
        else:
            seed = 0

    self.fdict['filter_n'] = N
    self.fdict['filter_lin'] = linear
    self.fdict['seed'] = seed

    if hasattr(self, 'data'):
        self.fdict['data'] = self.data
    elif 'data' in self.fdict.keys():
        self.data = self.fdict['data']

    self.Z = np.array(self.data)

    if obs_cov is None:
        if 'obs_cov' in self.fdict.keys():
            obs_cov = self.fdict['obs_cov']
        else:
            obs_cov = 1e-1

    if isinstance(obs_cov, float):
        obs_cov = self.create_obs_cov(obs_cov)

    self.fdict['obs_cov'] = obs_cov

    if not hasattr(self, 'sys'):
        self.get_sys(reduce_sys=True, verbose=verbose)

    self.preprocess(verbose=verbose > 1)

    self.create_filter(N=N, linear=linear, random_seed=seed)

    if 'filter_R' in self.fdict.keys():
        self.filter.R = self.fdict['filter_R']
    else:
        self.filter.R = obs_cov

    # dry run before the fun beginns
    if np.isinf(self.get_ll(constr_data=constr_data, verbose=verbose)):
        raise ValueError('[estimation:]'.ljust(
            15, ' ') + 'likelihood of initial values is zero.')

    print('[estimation:]'.ljust(15, ' ') +
          'Model operational. %s states, %s observables.' % (len(self.vv), len(self.observables)))

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

    print('[estimation:]'.ljust(15, ' ') +
          '%s priors detected. Adding parameters to the prior distribution.' % self.ndim)

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

                ll = self.get_ll(constr_data=constr_data, verbose=verbose)

                if verbose == 2:
                    print('[llike:]'.ljust(15, ' ') +
                          'Sample took '+str(np.round(time.time() - st, 3))+'s.')

                return ll

            except KeyboardInterrupt:
                raise

            except Exception as err:
                if verbose == 2:
                    print('[llike:]'.ljust(15, ' ') +
                          'Sample took '+str(np.round(time.time() - st, 3))+'s. (failure, error msg: %s)' % err)

                return -np.inf

    def lprior(pars):

        prior = 0
        for i, pl in enumerate(self.fdict['frozen_priors']):
            prior += pl.logpdf(pars[i])

        return prior

    def lprob(pars, linear=linear, verbose=verbose):

        return lprior(pars) + llike(pars, linear, verbose)

    global lprob_global
    lprob_global = lprob

    # also make functions accessible
    self.lprob = lprob
    self.lprior = lprior
    self.llike = llike


def get_init_par(self, nwalks, linear=False, use_top=None, distributed=False, ncores=None, verbose=False):

    if use_top is None:
        use_top = 0.

    if linear is None:
        linear = self.linear_filter

    if distributed:

        import pathos

        print()
        print('[estimation:]'.ljust(15, ' ') + 'finding initial values for mcmc (distributed over priors):')

        # globals are *evil*
        global lprob_global

        frozen_priors = self.fdict['frozen_priors']

        def runner(seed):

            np.random.seed(seed)

            draw_prob = -np.inf

            while np.isinf(draw_prob):
                nprr = np.random.randint
                pdraw = [pl.rvs(random_state=nprr(2**32-1)) for pl in frozen_priors]
                draw_prob = lprob_global(pdraw, linear, verbose)

            return np.array(pdraw)


        if ncores is None:
            ncores = pathos.multiprocessing.cpu_count()

        loc_pool = pathos.pools.ProcessPool(ncores)
        loc_pool.clear()

        pmap_sim = tqdm.tqdm(loc_pool.imap(runner, range(nwalks)), total=nwalks)

        return np.array(list(pmap_sim))

    else:

        if np.ndim(self.par_cand) > 1:

            ranking = (-self.fdict['swarms'][1][:, 0]).argsort()
            which = max(use_top*self.par_cand.shape[0], 1)
            par_cand = self.par_cand[ranking][:int(which)]

        else:
            par_cand = self.par_cand

        if np.ndim(par_cand) > 1:

            p0 = np.empty((nwalks, self.ndim))
            cand_dim = par_cand.shape[0]

            for i, w in enumerate(range(nwalks)):
                par = par_cand[i % cand_dim]
                p0[w, :] = par * (1+1e-3*np.random.randn())

            return p0

        elif nwalks > 1:
            return par_cand*(1+1e-3*np.random.randn(nwalks, self.ndim))
        else:
            return par_cand


def swarms(self, algos, linear=None, pop_size=100, ngen=500, mig_share=.1, seed=None, max_gen=None, use_ring=False, nlopt=True, broadcasting=True, ncores=None, crit_mem=.85, update_freq=None, verbose=False, debug=False):

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

    if max_gen is None:
        max_gen = ngen

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

            # if this is bobyqa or a friend
            if sname[:5] == 'NLopt':
                return sname.split(' ')[-1]

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

        prob = pg.problem(sfunc_inst)

        if nlopt and not seed:
            algo = pg.algorithm(pg.nlopt(solver="cobyla"))
            print('[swarms:]'.ljust(15, ' ') + 'On seed ' + str(seed)+' creating ' + algo.get_name())
            algo.extract(pg.nlopt).maxeval = pop_size
        elif nlopt and seed == 1:
            algo = pg.algorithm(pg.nlopt(solver="neldermead"))
            print('[swarms:]'.ljust(15, ' ') + 'On seed ' + str(seed)+' creating ' + algo.get_name())
            algo.extract(pg.nlopt).maxeval = pop_size
        else:
            random.seed(seed)
            algo = random.sample(algos, 1)[0]
            print('[swarms:]'.ljust(15, ' ') + 'On seed ' + str(seed)+' creating ' + algo.get_name())
            algo.set_seed(seed)

        pop = pg.population(prob, size=pop_size, seed=seed)
        ser_pop = dump_pop(pop)
        ser_algo = dill.dumps(algo)

        return ser_algo, ser_pop

    def evolve(ser_algo, ser_pop):

        algo = dill.loads(ser_algo)
        pop = load_pop(ser_pop)
        pop = algo.evolve(pop)

        return dill.dumps(algo), dump_pop(pop),

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    print('[swarms:]'.ljust(15, ' ') +
          'Number of evaluations per core is %sx the generation length.' % (ngen*pop_size/ncores))

    if not debug:
        pool = pathos.pools.ProcessPool(ncores)
        pool.clear()

    mig_abs = int(pop_size*mig_share)

    print('[swarms:]'.ljust(15, ' ') + 'Creating overlord of %s swarms...' % ncores)

    if debug:
        rests = [gen_pop(s, algos, pop_size) for s in range(ncores)]
        overlord = [Swarm(*res, s) for s, res in zip(range(ncores), rests)]
    else:
        rests = [pool.apipe(gen_pop, s, algos, pop_size)
                 for s in range(ncores)]
        overlord = [Swarm(*res.get(), s)
                    for s, res in zip(range(ncores), rests)]

        # better clear pool here already
        pool.clear()

    print('[swarms:]'.ljust(15, ' ') + 'Creating overlord of %s swarms...done.' % ncores)
    # print('done.')
    print('[swarms:]'.ljust(15, ' ') + 'Swarming out! Bzzzzz...')

    done = False
    best_x = None

    xsw = np.empty((ncores, self.ndim))
    fsw = np.empty((ncores, 1))
    nsw = np.empty((ncores, 1), dtype=object)

    pbar = tqdm.tqdm(total=ngen, dynamic_ncols=True)

    f_max = -np.inf

    if not debug and not verbose:
        np.warnings.filterwarnings('ignore')

    while not done:
        for s in overlord:

            if not use_ring:
                if not debug and not s.ready:
                    continue

                if pbar.n >= ngen:
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
                f_max_swarm = -fs[fas][0][0]
                if f_max_swarm > f_max:
                    f_max = f_max_swarm
                    x_max = xs[fas][0]
                    f_max_cnt = pbar.n
                    n_max = s.sname
                    if len(n_max) > 8:
                        n_max = s.sname[:5]+'_'+s.sname[-2:]

                pbar.set_description('ll: '+str(f_max_swarm.round(4)).rjust(12, ' ') + ('[%s/%s/%s]' %(f_max.round(4), n_max, f_max_cnt)).rjust(26, ' '))

                # keep us up to date
                if update_freq and pbar.n and not pbar.n % update_freq:

                    for i, sw in enumerate(overlord):
                        fsw[i, :] = -sw.pop[1].min()
                        xsw[i, :] = sw.pop[0][s.pop[1].argmin()]
                        nsw[i, :] = sw.sname

                    swarms = xsw, fsw, nsw.reshape(1, -1)
                    pbar.write(str(summary(
                        swarms, self['__data__']['estimation']['prior'], swarm_mode=True, show_priors=False).round(3)))

                if f_max_cnt < pbar.n - max_gen and f_max == f_max_swarm:
                    print('[swarms:]'.ljust(
                        15, ' ') + 'No improvement in the last %s generations, exiting...' % max_gen)
                    done = True
                    break

                # migrate the worst
                if best_x is not None and pbar.n < ngen:
                    for no, x, f in zip(fas[-mig_abs:], best_x, best_f):
                        s.pop[0][no] = x
                        s.pop[1][no] = f

                # save best for the next
                if broadcasting:
                    # this might include broadcasting the new candidates
                    xs = s.pop[0]
                    fs = s.pop[1]
                    fas = fs[:, 0].argsort()

                best_x = xs[fas][:mig_abs]
                best_f = fs[fas][:mig_abs]

            if pbar.n <= ngen:

                if crit_mem is not None:
                    # check if mem usage is above threshold
                    if psutil.virtual_memory()[2] > crit_mem:

                        pool.close()
                        print('[swarms:]'.ljust(15, ' ') + " Critical memory usage of "+str(
                            crit_mem)+"% reached, closing pools for maintenance...", end="", flush=True)
                        pool.join()
                        print('fixing...', end="", flush=True)
                        pool.restart()
                        print('done.')

                if not debug:
                    s.res = pool.apipe(evolve, s.algo, s.pop)
                else:
                    s.res = evolve(s.algo, s.pop)

        done = done or pbar.n >= ngen

    pool.terminate()
    pbar.close()

    hs = []

    for i, s in enumerate(overlord):
        fsw[i, :] = -s.pop[1].min()
        xsw[i, :] = s.pop[0][s.pop[1].argmin()]
        nsw[i, :] = s.sname
        hs.append(np.array(s.history))

    self.overlord = overlord
    self.par_cand = xsw

    self.fdict['ngen'] = ngen
    self.fdict['swarms'] = xsw, fsw, nsw.reshape(1, -1)
    self.fdict['swarm_history'] = hs

    return xsw


def mcmc(self, p0=None, nsteps=3000, nwalks=None, tune=None, seed=None, ncores=None, backend_file=None, linear=None, use_top=None, distr_init_chains=False, resume=False, update_freq=None, verbose=False, debug=False):

    import pathos
    import emcee

    self.fdict['use_top'] = use_top

    if not hasattr(self, 'ndim'):
        # if it seems to be missing, lets do it.
        # but without guarantee...
        self.prep_estim()

    if seed is None:
        seed = self.fdict['seed']

    if use_top is None:
        use_top = 0

    if tune is None:
        # self.tune = int(nsteps*4/5.)
        # 2/3 seems to be a better fit, given that we initialize at a good approximation of the posterior distribution
        self.tune = int(nsteps*2/3.)

    if update_freq is None:
        update_freq = int(nsteps/5.)

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

    if p0 is not None:
        pass
    elif resume:
        p0 = sampler.get_last_sample()
    else:
        p0 = get_init_par(self, nwalks, linear, use_top,
                          distr_init_chains, verbose)

    if not verbose:
        np.warnings.filterwarnings('ignore')

    if not verbose:
        pbar = tqdm.tqdm(total=nsteps, unit='sample(s)', dynamic_ncols=True)
        report = pbar.write
    else:
        report = print

    old_tau = np.inf

    for result in sampler.sample(p0, iterations=nsteps):

        cnt = sampler.iteration

        if cnt and update_freq and not cnt % update_freq:

            report('')
            if self.description is not None:
                report('[mcmc:]'.ljust(15, ' ') +
                       'Summary from last %s of %s iterations (%s):' % (update_freq, cnt, str(self.description)))

            else:
                report('[mcmc:]'.ljust(15, ' ') +
                       'Summary from last %s of %s iterations:' % (update_freq, cnt))

            sample = sampler.get_chain()

            tau = emcee.autocorr.integrated_time(sample, tol=0)
            max_tau = np.max(tau)
            dev_tau = np.max(np.abs(old_tau - tau)/tau)

            tau_sign = '>' if max_tau > cnt/50 else '<'
            dev_sign = '>' if dev_tau > .01 else '<'

            report(str(summary(sample, self.priors, tune=-update_freq).round(3)))
            report("Convergence stats: maxiumum tau is %s (%s%s) and change is %s (%s0.01)." % (
                max_tau.round(2), tau_sign, cnt/50, dev_tau.round(3), dev_sign))
            report("Mean likelihood is %s, mean acceptance fraction is %s." % (lprob_local(np.mean(
                sample[-update_freq:], axis=(0, 1))).round(3), np.mean(sampler.acceptance_fraction[-update_freq:]).round(2)))
            old_tau = tau

        if not verbose:
            pbar.update(1)

    pbar.close()

    if not verbose:
        np.warnings.filterwarnings('default')

    print("mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)))

    self.sampler = sampler

    return


def kdes(self, p0=None, nsteps=3000, nwalks=None, tune=None, seed=None, ncores=None, linear=None, use_top=None, distr_init_chains=False, resume=False, verbose=False, debug=False):

    import pathos
    import kombine
    from grgrlib.patches import kombine_run_mcmc

    kombine.Sampler.run_mcmc = kombine_run_mcmc

    self.fdict['use_top'] = use_top

    if not hasattr(self, 'ndim'):
        # if it seems to be missing, lets do it.
        # but without guarantee...
        self.prep_estim()

    if seed is None:
        seed = self.fdict['seed']

    np.random.seed(seed)

    if use_top is None:
        use_top = 0

    if tune is None:
        self.tune = None

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    if linear is None:
        linear = self.linear_filter

    if nwalks is None:
        nwalks = 120

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
        sampler = kombine.Sampler(nwalks, self.ndim, lprob_local)
    else:
        sampler = kombine.Sampler(
            nwalks, self.ndim, lprob_local, pool=loc_pool)

    if p0 is not None:
        pass
    elif resume:
        # should work, but not tested
        p0 = self.fdict['kdes_chain'][-1]
    else:
        p0 = get_init_par(self, nwalks, linear, use_top,
                          distr_init_chains, verbose)

    if not verbose:
        np.warnings.filterwarnings('ignore')

    if not verbose:
        pbar = tqdm.tqdm(total=nsteps, unit='sample(s)', dynamic_ncols=True)

    if nsteps < 500:
        nsteps_burnin = nsteps
        nsteps_mcmc = 0
    elif nsteps < 1000:
        nsteps_burnin = 500
        nsteps_mcmc = nsteps - nsteps_burnin
    else:
        nsteps_mcmc = 500
        nsteps_burnin = nsteps - nsteps_mcmc

    tune = max(500,nsteps_burnin)

    p, post, q = sampler.burnin(p0, max_steps=nsteps_burnin, pbar=pbar, verbose=verbose)

    if nsteps_mcmc:
        p, post, q = sampler.run_mcmc(nsteps_mcmc, pbar=pbar)

    acls = np.ceil(2/np.mean(sampler.acceptance[-tune:], axis=0) - 1).astype(int)
    samples = np.concatenate([sampler.chain[-tune::acl, c].reshape(-1, 2) for c, acl in enumerate(acls)])

    # samples = sampler.get_samples()

    kdes_chain = sampler.chain
    kdes_sample = samples.reshape(1, -1, self.ndim)

    self.kdes_chain = kdes_chain
    self.kdes_sample = kdes_sample
    self.fdict['tune'] = tune
    self.fdict['kdes_chain'] = kdes_chain
    self.fdict['kdes_sample'] = kdes_sample

    pbar.close()

    if not verbose:
        np.warnings.filterwarnings('default')

    self.sampler = sampler

    return


