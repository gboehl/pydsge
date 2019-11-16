#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import time
from .stats import get_prior, mc_mean, summary
from .core import get_par
from grgrlib.stuff import GPP
import tqdm
import cloudpickle as cpickle


def prep_estim(self, N=None, linear=None, load_R=False, seed=None, dispatch=False, constr_data=False, verbose=True, **filterargs):
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
    if np.isinf(self.get_ll(constr_data=constr_data, verbose=verbose > 3, dispatch=dispatch)):
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

    if 'frozen_prior' not in self.fdict.keys():

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
                    self.get_sys(par=par_active_lst, l_max=3, k_max=16, reduce_sys=True, verbose=verbose > 3)
                    self.filter.Q = self.QQ(self.par) @ self.QQ(self.par)
                else:
                    if not self.filter.name == 'KalmanFilter':
                        raise AttributeError('[estimation:]'.ljust(
                            15, ' ') + 'Missmatch between linearity choice (filter vs. lprob)')
                    self.get_sys(par=par_active_lst, l_max=1, k_max=0, reduce_sys=True, verbose=verbose > 3)
                    self.filter.F = self.linear_representation
                    self.filter.H = self.hx

                    CO = self.SIG @ self.QQ(self.par)
                    self.filter.Q = CO @ CO.T

                ll = self.get_ll(constr_data=constr_data,
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

    def lprob(par, linear=None, verbose=verbose, temp=1, lprob_seed='vec'):

        if linear is None:
            linear = linear_pa

        if verbose:
            st = time.time()

        if lprob_seed in ('vec', 'rand'):
            seed_loc = sum(p // 10**(int(np.log(abs(p))/np.log(10))-9) for p in par)
            if lprob_seed == 'rand':
                seed_loc += np.random.randint(2**32-2) 
            seed_loc = int(seed_loc) % (2**32 - 1)

        elif lprob_seed == 'set':
            seed_loc = seed
        else:
            raise NotImplementedError("`lprob_seed` must be one of `('vec', 'rand', 'set')`.")

        ll = llike(par, linear, verbose, seed_loc)*temp if temp else 0

        if np.isinf(ll):
            return ll

        ll += lprior(par)
        if verbose:
            print('[lprob:]'.ljust(15, ' ') + "Sample took %ss, ll is %s, temp is %s." %
                  (np.round(time.time() - st, 3), np.round(ll, 4), np.round(temp,3)))

        return ll

    ## old and evil way, kept for reference
    global lprob_global
    lprob_global = lprob

    ## make functions accessible
    self.lprob = lprob
    self.lprior = lprior
    self.llike = llike


def swarms(self, algos, linear=None, pop_size=100, ngen=500, mig_share=.1, seed=None, use_ring=False, nlopt=True, broadcasting=True, ncores=None, crit_mem=.85, autosave=100, update_freq=None, use_cloudpickle=False, verbose=False, debug=False):
    """Find mode using pygmo swarms.

    The interface partly replicates some features of the distributed island model because the original implementation has problems with the picklability of the DSGE class

    Parameters
    ----------
    algos : list
        List of initilized pygmo algorithms.
    linear : bool, optional
        Optimize linear model. Defaults to whether the filter object is linear.
    pop_size : int
        Size of each population. (Default: 100)
    ngen : in, optional
        Number of generations. Note that this runs *on top* of the generations defined in each algorithm. (default: 500)
    mig_share : floa, optional
        Percentage of solution candidates broadcasted and exchanged. (default: 0.1)
    use_ring : boo, optional
        Ordering of the execution of algorithm. If `False`, solutions will be evaluated and broadcasted whenever they are ready. The disadvantage is that results are not *exactly* reproducible since they depend on evaluation time. `False` is faster (if everything runs smoothly your results should not depend on random numbers). (default: False)
    nlopt : bool, optional
        Whether to let local optimizers run with the wolfs (and bees and ants). (default: True)
    broadcasting : boo, optional
        Whether to broadcast candidates to everybody or only to the next population. (default: True)
    """

    import pygmo as pg
    import pathos
    import random

    ## get the maximum generation len of all algos for nlopt methods
    if isinstance(nlopt, bool) and nlopt:
        maxalgogenlen = 1
        for algo in algos:
            st = algo.get_extra_info()
            if 'Generations' in st:
                genlen = int(st.split('\n')[0].split(' ')[-1])
                maxalgogenlen = max(maxalgogenlen, genlen)

        nlopt = maxalgogenlen*pop_size


    if crit_mem is not None:

        # TODO: one of the functions exposed by C++ leaks into memory...
        import psutil
        if crit_mem < 1:
            crit_mem *= 100

    if linear is None:
        linear = self.filter.name == 'KalmanFilter'

    if seed is None:
        seed = self.fdict['seed']

    if update_freq is None:
        update_freq = 0

    np.random.seed(seed)
    random.seed(seed)

    # globals are *evil*

    if not use_cloudpickle:
        global lprob_global
    else:
        lprob_dump = cpickle.dumps(self.lprob)
        lprob_global = cpickle.loads(lprob_dump)

    def lprob(par): return lprob_global(par, linear, verbose)

    sfunc_inst = GPP(lprob, self.fdict['prior_bounds'])

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

            algo = cpickle.loads(self.algo)
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
            algo.extract(pg.nlopt).maxeval = nlopt
        elif nlopt and seed == 1:
            algo = pg.algorithm(pg.nlopt(solver="neldermead"))
            print('[swarms:]'.ljust(15, ' ') + 'On seed ' +
                  str(seed)+' creating ' + algo.get_name())
            algo.extract(pg.nlopt).maxeval = nlopt
        else:
            random.seed(seed)
            algo = random.sample(algos, 1)[0]
            print('[swarms:]'.ljust(15, ' ') + 'On seed ' +
                  str(seed)+' creating ' + algo.get_name())
            algo.set_seed(seed)

        pop = pg.population(prob, size=pop_size, seed=seed)
        ser_pop = dump_pop(pop)
        ser_algo = cpickle.dumps(algo)

        return ser_algo, ser_pop

    def evolve(ser_algo, ser_pop):

        algo = cpickle.loads(ser_algo)
        pop = load_pop(ser_pop)
        pop = algo.evolve(pop)

        return cpickle.dumps(algo), dump_pop(pop),

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

                self.fdict['swarm_history'] = np.array(f_max_hist).reshape(
                    1, -1), np.array(x_max_hist), np.array(name_max_hist).reshape(1, -1)

                fas = fsw[:, 0].argmax()
                self.fdict['swarms_mode_x'] = xsw[fas]
                self.fdict['swarms_mode_f'] = fsw[fas]

                if 'mode_f' in self.fdict.keys() and fsw[fas] < self.fdict['mode_f']:
                    if done:
                        print('[swarms:]'.ljust(
                            15, ' ') + " New mode of %s is below old mode of %s. Rejecting..." % (fsw[fas], self.fdict['mode_f']))
                else:
                    self.fdict['mode_x'] = xsw[fas]
                    self.fdict['mode_f'] = fsw[fas]

                for i, s in enumerate(overlord):
                    fsw[i, :] = -s.pop[1].min()
                    xsw[i, :] = s.pop[0][s.pop[1].argmin()]
                    nsw[i, :] = s.sname

                self.fdict['ngen'] = ngen
                self.fdict['swarms'] = xsw, fsw, nsw.reshape(1, -1)

                if autosave:
                    self.save(verbose=verbose)

    pbar.close()

    return xsw


def mcmc(self, p0=None, nsteps=3000, nwalks=None, tune=None, moves=None, temp=False, seed=None, ncores=None, backend=True, linear=None, distr_init_chains=False, resume=False, update_freq=None, lprob_seed=None, use_cloudpickle=False, verbose=False, debug=False):

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
        # self.tune = int(nsteps*1/5.)
        self.tune = int(nsteps*2/5.)

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
    if not use_cloudpickle:
        global lprob_global
    else:
        lprob_dump = cpickle.dumps(self.lprob)
        lprob_global = cpickle.loads(lprob_dump)

    if isinstance(temp, bool) and not temp:
        temp = 1

    def lprob(par): return lprob_global(par, linear, verbose, temp, lprob_seed or 'vec')

    loc_pool = pathos.pools.ProcessPool(ncores)
    loc_pool.clear()

    if debug:
        sampler = emcee.EnsembleSampler(nwalks, self.ndim, lprob)
    else:
        sampler = emcee.EnsembleSampler(nwalks, self.ndim, lprob, moves=moves, pool=loc_pool, backend=backend)

    self.sampler = sampler
    self.temp = temp

    if p0 is not None:
        pass
    elif resume:
        p0 = sampler.get_last_sample()
    elif temp < 1:
        p0 = get_par(self, 'prior_mean', asdict=False, full=False, nsample=nwalks, verbose=verbose)
    else:
        p0 = get_par(self, 'best', asdict=False, full=False, nsample=nwalks, verbose=verbose)

    if not verbose:
        np.warnings.filterwarnings('ignore')

    if verbose > 2:
        report = print
    else:
        pbar = tqdm.tqdm(total=nsteps, unit='sample(s)', dynamic_ncols=True)
        report = pbar.write

    old_tau = np.inf

    for result in sampler.sample(p0, iterations=nsteps):

        cnt = sampler.iteration

        if not verbose:
            pbar.set_description('[MAF: %s]' % (
                np.mean(sampler.acceptance_fraction[-update_freq:]).round(3)))

        if cnt and update_freq and not cnt % update_freq:

            prnttup = '[mcmc:]'.ljust(15, ' ') + "Summary from last %s of %s iterations" % (update_freq, cnt)

            if temp < 1:
                prnttup += ' with temp of %s' %np.round(temp, 3)

            if self.description is not None:
                prnttup += ' (%s)' %str(self.description)

            prnttup += ':'

            report(prnttup)

            sample = sampler.get_chain()

            tau = emcee.autocorr.integrated_time(sample, tol=0)
            min_tau = np.min(tau).round(2)
            max_tau = np.max(tau).round(2)
            dev_tau = np.max(np.abs(old_tau - tau)/tau)

            tau_sign = '>' if max_tau > cnt/50 else '<'
            dev_sign = '>' if dev_tau > .01 else '<'


            self.mcmc_summary(tune=update_freq, calc_mdd=False, calc_ll_stats=True, out=lambda x: report(str(x)))

            report("Convergence stats: tau is in (%s,%s) (%s%s) and change is %s (%s0.01)." % (
                min_tau, max_tau, tau_sign, cnt/50, dev_tau.round(3), dev_sign))

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

    log_probs = sampler.get_log_prob()[-self.tune:]
    chain = sampler.get_chain()[-self.tune:]
    chain = chain.reshape(-1, chain.shape[-1])

    arg_max = log_probs.argmax()
    mode_f = log_probs.flat[arg_max]
    mode_x = chain[arg_max].flatten()

    self.fdict['mcmc_mode_x'] = mode_x
    self.fdict['mcmc_mode_f'] = mode_f

    if temp == 1:
        if 'mode_f' in self.fdict.keys() and mode_f < self.fdict['mode_f']:
            print('[mcmc:]'.ljust(15, ' ') + "New mode of %s is below old mode of %s. Rejecting..." %
                  (mode_f, self.fdict['mode_f']))
        else:
            self.fdict['mode_x'] = mode_x
            self.fdict['mode_f'] = mode_f

    self.sampler = sampler

    return


def tmcmc(self, ntemps, nsteps, nwalks, update_freq=False, tempscale=2, verbose=False, **mcmc_args):
    """Run Tempered Ensemble MCMC
    """

    pars = get_par(self, 'prior_mean', asdict=False, full=False, nsample=nwalks, verbose=verbose)

    for tmp in np.linspace(0,1,ntemps)**tempscale:

        if tmp:
            print('[tmcmc:]'.ljust(15, ' ') + "Increasing tempearture to %s." %np.round(tmp, 3))


        self.mcmc(p0=pars, nsteps=nsteps, nwalks=nwalks, temp=tmp, update_freq=update_freq, verbose=verbose, backend=False, **mcmc_args)

        pars = self.get_chain()[-1]
        self.temp = tmp

        self.mcmc_summary(tune=int(nsteps/10), calc_mdd=False, calc_ll_stats=True)

    return pars


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

    if not use_cloudpickle:
        # globals are *evil*
        global lprob_global
    else:
        lprob_dump = cpickle.dumps(self.lprob)
        lprob_global = cpickle.loads(lprob_dump)

    def lprob(par): return lprob_global(par, linear, verbose)

    loc_pool = pathos.pools.ProcessPool(ncores)
    loc_pool.clear()

    if debug:
        sampler = kombine.Sampler(nwalks, self.ndim, lprob)
    else:
        sampler = kombine.Sampler(
            nwalks, self.ndim, lprob, pool=loc_pool)

    if p0 is not None:
        pass
    elif resume:
        # should work, but not tested
        p0 = self.fdict['kdes_chain'][-1]
    else:
        p0 = get_par(self, 'best', asdict=False, nsample=nwalks, verbose=verbose)

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


def cmaes(self, p0=None, pop_size=None, seeds=3, initseed=None, stagtol=150, ftol=5e-4, xtol=2e-4, linear=None, lprob_seed=None, use_cloudpickle=False, ncores=None, verbose=True, debug=False):
    """Find mode using CMA-ES.

    The interface partly replicates some features of the distributed island model because the original implementation has problems with the picklability of the DSGE class

    Parameters
    ----------
    pop_size : int
        Size of each population. (Default: number of dimensions)
    seeds : in, optional
        Number of different seeds tried. (Default: 3)
    """

    import cma
    import pathos

    np.random.seed(initseed or self.fdict['seed'])

    if isinstance(seeds, int):
        seeds = np.random.randint(2**32-2, size=seeds)

    ncores = pathos.multiprocessing.cpu_count()

    bnd = np.array(self.fdict['prior_bounds'])
    p0 = p0 or get_par(self, 'prior_mean', full=False, asdict=False) 
    p0 = (p0 - bnd[0])/(bnd[1] - bnd[0])

    pop_size = pop_size or ncores*np.ceil(len(p0)/ncores)

    opt_dict = { 
        'popsize': pop_size, 
        'tolstagnation': stagtol, 
        'tolfun': ftol, 
        'tolx': xtol, 
        'bounds': [0,1], 
        'verbose': verbose }

    if not use_cloudpickle:
        global lprob_global
    else:
        lprob_dump = cpickle.dumps(self.lprob)
        lprob_global = cpickle.loads(lprob_dump)

    def lprob(par): return lprob_global(par, linear=linear, lprob_seed=lprob_seed or 'rand')
    lprob_scaled = lambda x: -lprob((bnd[1] - bnd[0])*x + bnd[0])
    nhandler = None if lprob_seed == 'set' else cma.NoiseHandler(len(p0), parallel=True)

    if not debug:
        pool = pathos.pools.ProcessPool(ncores)
        pool.clear()
        mapper = pool.imap
    else:
        mapper = map

    if not debug and verbose < 2:
        np.warnings.filterwarnings('ignore')

    lprob_pooled = lambda X: list(mapper(lprob_scaled, list(X)))

    f_max = -np.inf

    print('[cma-es:]'.ljust(15, ' ') + 'Starting mode search over %s seeds...' %(seeds if isinstance(seeds, int) else len(seeds)))

    f_hist = []
    x_hist = []
    mean_hist = []
    std_hist = []
    
    for s in seeds:

        opt_dict['seed'] = s
        res = cma.fmin(None, p0, .25, parallel_objective=lprob_pooled, options=opt_dict, noise_handler=nhandler)

        x_scaled = res[0] * (bnd[1] - bnd[0]) + bnd[0]
        mean_scaled = res[5] * (bnd[1] - bnd[0]) + bnd[0]
        std_scaled = res[6] * (bnd[1] - bnd[0]) 
        f_hist.append(-res[1])
        x_hist.append(x_scaled)
        mean_hist.append(mean_scaled)
        std_hist.append(std_scaled)

        if -res[1] > f_max:

            f_max = -res[1]
            x_max_scaled = x_scaled
            if verbose:
                print('[cma-es:]'.ljust(15, ' ') + 'Updating best solution to %s at seed %s.' %(np.round(f_max, 4), s))

        elif verbose:
            print('[cma-es:]'.ljust(15, ' ') + 'Current solution of %s rejected at seed %s.' %(np.round(-res[1], 4), s))

        if verbose:
            from .clsmethods import cmaes_summary
            cmaes_summary(self, data=(f_hist, x_hist, mean_hist, std_hist))
            print('')

    np.warnings.filterwarnings('default')

    self.fdict['cmaes_mode_x'] = x_max_scaled
    self.fdict['cmaes_mode_f'] = f_max
    self.fdict['cmaes_history'] = f_hist, x_hist, seeds

    if 'mode_f' in self.fdict.keys() and f_max < self.fdict['mode_f']:
        if done:
            print('[swarms:]'.ljust(
                15, ' ') + " New mode of %s is below old mode of %s. Rejecting..." % (f_max, self.fdict['mode_f']))
    else:
        self.fdict['mode_x'] = x_max_scaled
        self.fdict['mode_f'] = f_max

    return x_max_scaled
