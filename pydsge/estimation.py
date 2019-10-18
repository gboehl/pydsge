#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
import os
import time
from .stats import get_prior, mc_mean, summary, pmdm_report
from grgrlib.stuff import GPP, map2arr
import scipy.optimize as so
import tqdm


def prep_estim(self, N=None, linear=None, load_R=False, seed=None, dispatch=False, constr_data=False, verbose=True):
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
    dispatch : bool, optional
        Whether to apply the constraint to the data as well. Defaults to False.
    verbose : bool/int, optional
        Whether display messages (and to which degree). Defaults to True.
    """

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

    if not hasattr(self, 'sys'):
        self.get_sys(reduce_sys=True, verbose=verbose > 1)
    if not hasattr(self, 'precalc_mat'):
        self.preprocess(verbose=verbose > 1)

    self.create_filter(
        N=N, ftype='KalmanFilter' if linear else None, random_seed=seed)

    if 'filter_R' in self.fdict.keys():
        self.filter.R = self.fdict['filter_R']
    elif load_R:
        raise AttributeError('[estimation:]'.ljust(
            15, ' ') + "`filter.R` not in `fdict`.")

    # dry run before the fun beginns
    if np.isinf(self.get_ll(constr_data=constr_data, verbose=verbose > 1, dispatch=dispatch)):
        raise ValueError('[estimation:]'.ljust(
            15, ' ') + 'likelihood of initial values is zero.')

    if verbose:
        print('[estimation:]'.ljust(15, ' ') + 'Model operational. %s states, %s observables.' %
              (len(self.vv), len(self.observables)))

    prior = self.prior
    par_fix = self.par_fix
    prior_arg = self.prior_arg

    # add to class so that it can be stored later
    self.fdict['prior_names'] = [pp for pp in prior.keys()]

    self.ndim = len(prior.keys())

    if 'frozen_prior' not in self.fdict.keys():

        pfrozen, pinitv, bounds = get_prior(prior)
        self.fdict['frozen_prior'] = pfrozen
        self.fdict['prior_bounds'] = bounds
        self.fdict['init_value'] = pinitv

    if verbose:
        print('[estimation:]'.ljust(
            15, ' ') + '%s priors detected. Adding parameters to the prior distribution.' % self.ndim)

    def llike(parameters, linear, verbose):

        random_state = np.random.get_state()
        with warnings.catch_warnings(record=True):
            try:
                warnings.filterwarnings('error')

                np.random.seed(seed)

                par_fix[prior_arg] = parameters
                par_active_lst = list(par_fix)

                self.get_sys(par=par_active_lst, reduce_sys=True,
                             verbose=verbose > 2)

                if not linear:
                    if self.filter.name == 'KalmanFilter':
                        raise AttributeError('[estimation:]'.ljust(
                            15, ' ') + 'Missmatch between linearity choice (filter vs. lprob)')
                    # these max vals should be sufficient given we're only dealing with stochastic linearization
                    self.preprocess(l_max=3, k_max=16, verbose=verbose > 2)
                    self.filter.Q = self.QQ(self.par) @ self.QQ(self.par)
                else:
                    if not self.filter.name == 'KalmanFilter':
                        raise AttributeError('[estimation:]'.ljust(
                            15, ' ') + 'Missmatch between linearity choice (filter vs. lprob)')
                    self.preprocess(l_max=1, k_max=0, verbose=False)
                    self.filter.F = self.linear_representation
                    self.filter.H = self.hx

                    CO = self.SIG @ self.QQ(self.par)
                    self.filter.Q = CO @ CO.T

                ll = self.get_ll(constr_data=constr_data,
                                 verbose=verbose > 2, dispatch=dispatch)

                np.random.set_state(random_state)
                return ll

            except KeyboardInterrupt:
                raise

            except Exception as err:
                if verbose:
                    print('[llike:]'.ljust(15, ' ') +
                          'Failure. Error msg: %s' % err)
                    if verbose > 1:
                        print(self.get_calib(parname='estim'))

                np.random.set_state(random_state)
                return -np.inf

    def lprior(par):

        prior = 0
        for i, pl in enumerate(self.fdict['frozen_prior']):
            prior += pl.logpdf(par[i])

        return prior

    linear_pa = linear

    def lprob(par, linear=None, verbose=verbose):

        if linear is None:
            linear = linear_pa

        if verbose:
            st = time.time()

        ll = llike(par, linear, verbose)

        if np.isinf(ll):
            return ll

        ll += lprior(par)
        if verbose:
            print('[lprob:]'.ljust(15, ' ') + "Sample took %ss, ll is %s." %
                  (np.round(time.time() - st, 3), np.round(ll, 4)))

        return ll

    global lprob_global
    lprob_global = lprob

    # also make functions accessible
    self.lprob = lprob
    self.lprior = lprior
    self.llike = llike


def get_par(self, which=None, nsample=1, seed=None, ncores=None, verbose=False):
    """Get parameters

    Parameters
    ----------
    which : str
        Can be one of {'prior', 'mode', 'calib', 'pmean', 'init'}. 
    nsample : int
        Size of the prior sample
    ncores : int
        Number of cores used for prior sampling. Defaults to the number of available processors

    Returns
    -------
    array
        Numpy array of parameters
    """

    if which is None:
        which = 'mode' if 'mode_x' in self.fdict.keys() else 'init'

    if which is 'prior':

        import pathos

        if seed is None:
            seed = 0

        if verbose:
            print('[estimation:]'.ljust(15, ' ') +
                  'finding initial values for mcmc (distributed over prior):')

        if not hasattr(self, 'ndim'):
            self.prep_estim(load_R=True, verbose=verbose)

        # globals are *evil*
        global lprob_global

        frozen_prior = self.fdict['frozen_prior']

        def runner(locseed):

            if seed is not None:
                np.random.seed(seed+locseed)

            draw_prob = -np.inf

            while np.isinf(draw_prob):
                with warnings.catch_warnings(record=False):
                    try:
                        warnings.filterwarnings('error')
                        rst = np.random.randint(2**16)
                        pdraw = [pl.rvs(random_state=rst) for pl in frozen_prior]
                        draw_prob = lprob_global(pdraw, None, verbose)
                    except:
                        pass

            return pdraw

        if ncores is None:
            ncores = pathos.multiprocessing.cpu_count()

        loc_pool = pathos.pools.ProcessPool(ncores)
        loc_pool.clear()

        pmap_sim = tqdm.tqdm(loc_pool.imap(runner, range(nsample)), total=nsample)

        return map2arr(pmap_sim)

    if which is 'mode':
        par_cand = self.fdict['mode_x']
    if which is 'calib':
        par_cand = self.par_fix[self.prior_arg]
    if which is 'pmean':
        par_cand = [prior[pp][1] for pp in prior.keys()]
    if which is 'init':
        par_cand = self.fdict['init_value']
        for i in range(self.ndim):
            if par_cand[i] is None:
                par_cand[i] = self.par_fix[self.prior_arg][i]

    try:
        return par_cand*(1 + 1e-3*np.random.randn(nsample, self.ndim)*(nsample > 1))
    except UnboundLocalError:
        raise KeyError("`which` must be in {'prior', 'mode', 'calib', 'pmean', 'init'")


def swarms(self, algos, linear=None, pop_size=100, ngen=500, mig_share=.1, seed=None, max_gen=None, initialize_x0=True, use_ring=False, nlopt=True, broadcasting=True, ncores=None, crit_mem=.85, autosave=100, update_freq=None, verbose=False, debug=False):

    import pygmo as pg
    import dill
    import pathos
    import random

    if crit_mem is not None:

        # TODO: one of the functions exposed by C++ leaks into memory...
        import psutil
        if crit_mem < 1:
            crit_mem *= 100

    if linear is None:
        linear = self.filter.name == 'KalmanFilter'

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
            print('[swarms:]'.ljust(15, ' ') + 'On seed ' +
                  str(seed)+' creating ' + algo.get_name())
            algo.extract(pg.nlopt).maxeval = pop_size
        elif nlopt and seed == 1:
            algo = pg.algorithm(pg.nlopt(solver="neldermead"))
            print('[swarms:]'.ljust(15, ' ') + 'On seed ' +
                  str(seed)+' creating ' + algo.get_name())
            algo.extract(pg.nlopt).maxeval = pop_size
        else:
            random.seed(seed)
            algo = random.sample(algos, 1)[0]
            print('[swarms:]'.ljust(15, ' ') + 'On seed ' +
                  str(seed)+' creating ' + algo.get_name())
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

    print('[swarms:]'.ljust(15, ' ') +
          'Creating overlord of %s swarms...' % ncores)

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

    print('[swarms:]'.ljust(15, ' ') +
          'Creating overlord of %s swarms...done.' % ncores)
    print('[swarms:]'.ljust(15, ' ') + 'Swarming out! Bzzzzz...')

    done = False
    best_x = None

    xsw = np.empty((ncores, self.ndim))
    fsw = np.empty((ncores, 1))
    nsw = np.empty((ncores, 1), dtype=object)

    pbar = tqdm.tqdm(total=ngen, dynamic_ncols=True)

    f_max = -np.inf
    f_max_cnt = 0
    f_max_hist = []
    x_max_hist = []
    name_max_hist = []

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

                s.ncalls += 1
                pbar.update()

                # keep us informed
                f_max_swarm = -fs[fas][0][0]

                if f_max_swarm > f_max:
                    f_max = f_max_swarm
                    x_max = xs[fas][0]
                    f_max_cnt = pbar.n
                    name_max = s.sname

                if not np.isinf(f_max):
                    f_max_hist.append(f_max)
                    x_max_hist.append(x_max)
                    name_max_hist.append(name_max)

                    name_len = 25 - 9 - \
                        len(str(int(f_max))) - len(str(f_max_cnt))

                    try:
                        name0, name1 = name_max.split('_')
                        sname = name0[:name_len-len(name1)-1] + '_' + name1
                    except:
                        sname = name_max[:name_len]

                    pbar.set_description('ll: '+str(f_max_swarm.round(4)).rjust(11, ' ') + (
                        '[%s/%s/%s]' % (f_max.round(4), sname, f_max_cnt)).rjust(26, ' '))

                # keep us up to date
                if update_freq and pbar.n and not pbar.n % update_freq:

                    for i, sw in enumerate(overlord):
                        fsw[i, :] = -sw.pop[1].min()
                        xsw[i, :] = sw.pop[0][s.pop[1].argmin()]
                        nsw[i, :] = sw.sname

                    swarms = xsw, fsw, nsw.reshape(1, -1)
                    pbar.write(str(summary(
                        swarms, self['__data__']['estimation']['prior'], swarm_mode=True).round(3)))

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

            done |= pbar.n >= ngen

            if done or (pbar.n and not pbar.n % autosave):

                self.fdict['swarm_history'] = np.array(f_max_hist).reshape(1, -1), np.array(x_max_hist), np.array(name_max_hist).reshape(1, -1)

                fas = fsw[:, 0].argmax()
                self.fdict['swarms_x'] = xsw[fas]
                self.fdict['swarms_f'] = fsw[fas]

                if 'mode_f' in self.fdict.keys() and fsw[fas] < self.fdict['mode_f']:
                    if done:
                        print('[swarms:]'.ljust(15, ' ') + " New mode of %s is below old mode of %s. Rejecting..." %(fsw[fas], self.fdict['mode_f']))
                else:
                    self.fdict['mode_x'] = xsw[fas]
                    self.fdict['mode_f'] = fsw[fas]

                if autosave:
                    self.save(verbose=verbose)

    pool.terminate()
    pbar.close()

    for i, s in enumerate(overlord):
        fsw[i, :] = -s.pop[1].min()
        xsw[i, :] = s.pop[0][s.pop[1].argmin()]
        nsw[i, :] = s.sname

    self.overlord = overlord

    self.fdict['ngen'] = ngen
    self.fdict['swarms'] = xsw, fsw, nsw.reshape(1, -1)

    return xsw


def mcmc(self, p0=None, nsteps=3000, nwalks=None, tune=None, seed=None, ncores=None, backend=True, linear=None, distr_init_chains=False, resume=False, update_freq=None, verbose=False, debug=False):

    import pathos
    import emcee

    if not hasattr(self, 'ndim'):
        # if it seems to be missing, lets do it.
        # but without guarantee...
        self.prep_estim(load_R=True)

    if seed is None:
        seed = self.fdict['seed']

    self.tune = tune
    if tune is None:
        # self.tune = int(nsteps*4/5.)
        # 2/3 seems to be a better fit, given that we initialize at a good approximation of the posterior distribution
        self.tune = int(nsteps*2/3.)

    if update_freq is None:
        update_freq = int(nsteps/5.)

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    if linear is None:
        linear = self.filter.name == 'KalmanFilter'

    if backend:
        if isinstance(backend, str):
            self.backend_file = backend
        if 'backend_file' in self.fdict.keys():
            self.backend_file = str(self.fdict['backend_file'])
        else:
            self.backend_file = self.path + self.name + '_sampler.h5'

        backend = emcee.backends.HDFBackend(self.backend_file)

        if not resume:
            backend.reset(nwalks, self.ndim)
    else:
        backend = None

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
        p0 = get_par(self, nsample=nwalks, verbose=verbose)

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
        if not verbose:
            pbar.set_description('[MAF: %s]' % (
                np.mean(sampler.acceptance_fraction[-update_freq:]).round(3)))

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
            min_tau = np.min(tau).round(2)
            max_tau = np.max(tau).round(2)
            dev_tau = np.max(np.abs(old_tau - tau)/tau)

            tau_sign = '>' if max_tau > cnt/50 else '<'
            dev_sign = '>' if dev_tau > .01 else '<'

            report(str(summary(sample, self.prior, tune=-update_freq).round(3)))
            report("Convergence stats: tau is in (%s,%s) (%s%s) and change is %s (%s0.01)." % (
                min_tau, max_tau, tau_sign, cnt/50, dev_tau.round(3), dev_sign))
            report("Likelihood at mean is %s, mean acceptance fraction is %s." % (lprob_local(np.mean(
                sample[-update_freq:], axis=(0, 1))).round(3), np.mean(sampler.acceptance_fraction[-update_freq:]).round(2)))

        if cnt and update_freq and not (cnt+1) % update_freq:
            sample = sampler.get_chain()
            old_tau = emcee.autocorr.integrated_time(sample, tol=0)

        if not verbose:
            pbar.update(1)

    pbar.close()

    if not verbose:
        np.warnings.filterwarnings('default')

    print("mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)))

    log_probs = sampler.get_log_prob()[self.tune:]
    chain = sampler.get_chain()[self.tune:]
    chain = chain.reshape(-1, chain.shape[-1])

    arg_max = log_probs.argmax()
    mode_f = log_probs.flat[arg_max]
    mode_x = chain[arg_max]

    self.fdict['mcmc_mode_x'] = mode_x
    self.fdict['mcmc_mode_f'] = mode_f

    if 'mode_f' in self.fdict.keys() and mode_f < self.fdict['mode_f']:
        print('[mcmc:]'.ljust(15, ' ') + "New mode of %s is below old mode of %s. Rejecting..." %
              (mode_f, self.fdict['mode_f']))
    else:
        self.fdict['mode_x'] = mode_x
        self.fdict['mode_f'] = mode_f

    self.sampler = sampler

    return


def kdes(self, p0=None, nsteps=3000, nwalks=None, tune=None, seed=None, ncores=None, linear=None, distr_init_chains=False, resume=False, verbose=False, debug=False):

    import pathos
    import kombine
    from grgrlib.patches import kombine_run_mcmc

    kombine.Sampler.run_mcmc = kombine_run_mcmc

    if not hasattr(self, 'ndim'):
        # if it seems to be missing, lets do it.
        # but without guarantee...
        self.prep_estim(load_R=True)

    if seed is None:
        seed = self.fdict['seed']

    np.random.seed(seed)

    if tune is None:
        self.tune = None

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    if linear is None:
        linear = self.filter.name == 'KalmanFilter'

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
        p0 = get_par(self, nsample=nwalks, verbose=verbose)

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

    tune = max(500, nsteps_burnin)

    p, post, q = sampler.burnin(
        p0, max_steps=nsteps_burnin, pbar=pbar, verbose=verbose)

    if nsteps_mcmc:
        p, post, q = sampler.run_mcmc(nsteps_mcmc, pbar=pbar)

    acls = np.ceil(
        2/np.mean(sampler.acceptance[-tune:], axis=0) - 1).astype(int)
    samples = np.concatenate(
        [sampler.chain[-tune::acl, c].reshape(-1, 2) for c, acl in enumerate(acls)])

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

    log_probs = sampler.get_log_prob()[self.tune:]
    chain = sampler.get_chain()[self.tune:]
    chain = chain.reshape(-1, chain.shape[-1])

    arg_max = log_probs.argmax()
    mode_f = log_probs.flat[arg_max]
    mode_x = chain[arg_max]

    self.fdict['kombine_mode_x'] = mode_x
    self.fdict['kombine_mode_f'] = mode_f

    if 'mode_f' in self.fdict.keys() and mode_f < self.fdict['mode_f']:
        print('[kombine:]'.ljust(15, ' ') + "New mode of %s is below old mode of %s. Rejecting..." %
              (mode_f, self.fdict['mode_f']))
    else:
        self.fdict['mode_x'] = mode_x
        self.fdict['mode_f'] = mode_x

    self.sampler = sampler

    return
