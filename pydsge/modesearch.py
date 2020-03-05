#!/bin/python
# -*- coding: utf-8 -*-

import os
import time
import tqdm
import numpy as np
from .core import get_par
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


def nlopt(self, p0=None, linear=None, maxfev=None, method=None, tol=1e-2, update_freq=None, verbose=False):

    from pydsge.estimation import GPP, get_par
    import pygmo as pg

    print('[pmdm:]'.ljust(15, ' ') + "WARNING: I have not used this function for quite a while, it is unmaintained and probably malfunctioning! `cmaes` is likely to do a better job.")

    if linear is None:
        linear = self.linear_filter

    def lprob(x): return self.lprob(x, linear=linear, verbose=verbose)

    sfunc_inst = GPP(lprob, self.fdict['prior_bounds'])

    if p0 is None:
        p0 = get_par('best', self, linear=linear, verbose=verbose, full=False)

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
    self.fdict['nlopt_x'] = pop.champion_x
    self.fdict['nlopt_f'] = pop.champion_f

    if 'mode_f' in self.fdict.keys() and pop.champion_f < self.fdict['mode_f']:
        print('[pmdm:]'.ljust(15, ' ') + " New mode of %s is below old mode of %s. Rejecting..." %
              (pop.champion_f, self.fdict['mode_f']))
    else:
        self.fdict['mode_x'] = pop.champion_x
        self.fdict['mode_f'] = pop.champion_f

    return pop.champion_x


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

    import pathos
    import random
    import cloudpickle as cpickle
    import pygmo as pg
    from grgrlib.core import GPP

    print('[pmdm:]'.ljust(15, ' ') + "WARNING: I have not used this function for quite a while, it is unmaintained and probably malfunctioning! `cmaes` is likely to do a better job.")

    # get the maximum generation len of all algos for nlopt methods
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
        import cloudpickle as cpickle
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
    if not debug:
        pool.close()

    return xsw


def cmaes2(self, p0=None, sigma=None, pop_size=None, seeds=3, seed=None, stagtol=150, ftol=5e-4, xtol=2e-4, burnin=False, linear=None, lprob_seed=None, use_cloudpickle=False, ncores=None, cma_callback=None, verbose=True, debug=False):
    """Find mode using CMA-ES from pycma.

    Parameters
    ----------
    pop_size : int
        Size of each population. (Default: number of dimensions)
    seeds : in, optional
        Number of different seeds tried. (Default: 3)
    """

    print('[pmdm:]'.ljust(15, ' ') + "WARNING: I have not used this function for quite a while, it is unmaintained and probably malfunctioning! `cmaes` is likely to do a better job.")

    import cma
    import pathos

    np.random.seed(seed or self.fdict['seed'])

    if isinstance(seeds, int):
        if burnin:
            seeds += 1
        seeds = np.random.randint(2**31, size=seeds)  # win explodes with 2**32

    ncores = pathos.multiprocessing.cpu_count()

    bnd = np.array(self.fdict['prior_bounds'])
    p0 = get_par(self, 'prior_mean', full=False,
                 asdict=False) if p0 is None else p0
    p0 = (p0 - bnd[0])/(bnd[1] - bnd[0])
    sigma = sigma or .2

    pop_size = pop_size or ncores*np.ceil(len(p0)/ncores)

    opt_dict = {
        'popsize': ncores if burnin else pop_size,
        'tolstagnation': stagtol,
        'tolfun': ftol,
        'tolx': xtol,
        'bounds': [0, 1],
        'verb_disp': 2000,
        'verbose': verbose}

    if not use_cloudpickle:
        global lprob_global
    else:
        import cloudpickle as cpickle
        lprob_dump = cpickle.dumps(self.lprob)
        lprob_global = cpickle.loads(lprob_dump)

    def lprob(par): return lprob_global(
        par, linear=linear, lprob_seed=lprob_seed or 'set')

    def lprob_scaled(x): return -lprob((bnd[1] - bnd[0])*x + bnd[0])
    nhandler = None if lprob_seed == 'set' else cma.NoiseHandler(
        len(p0), parallel=True)

    if not debug:
        pool = pathos.pools.ProcessPool(ncores)
        pool.clear()
        mapper = pool.imap
    else:
        mapper = map

    if not debug and verbose < 2:
        np.warnings.filterwarnings('ignore')

    def lprob_pooled(X): return list(mapper(lprob_scaled, list(X)))

    f_max = -np.inf

    print('[cma-es:]'.ljust(15, ' ') + 'Starting mode search over %s seeds...' %
          (seeds if isinstance(seeds, int) else len(seeds)))

    f_hist = []
    x_hist = []

    for s in seeds:

        opt_dict['seed'] = s
        res = cma.fmin(None, p0, sigma, parallel_objective=lprob_pooled,
                       options=opt_dict, noise_handler=nhandler, callback=cma_callback)

        repair = res[-2].boundary_handler.repair
        x_scaled = res[0] * (bnd[1] - bnd[0]) + bnd[0]
        f_hist.append(-res[1])
        x_hist.append(x_scaled)

        check_bnd = np.bitwise_or(res[0] < 1e-3, res[0] > 1-1e-3)

        if -res[1] < f_max:
            print('[cma-es:]'.ljust(15, ' ') +
                  'Current solution of %s rejected at seed %s.' % (np.round(-res[1], 4), s))

        elif check_bnd.any():
            print('[cma-es:]'.ljust(15, ' ') + 'Current solution of %s rejected at seed %s because %s is at the bound.' %
                  (np.round(-res[1], 4), s, np.array(self.prior_names)[check_bnd]))

        else:
            f_max = -res[1]
            x_max_scaled = x_scaled
            if verbose:
                print('[cma-es:]'.ljust(15, ' ') +
                      'Updating best solution to %s at seed %s.' % (np.round(f_max, 4), s))

        if verbose:
            from .clsmethods import mode_summary
            mode_summary(self, data_cmaes=(f_hist, x_hist))
            print('')

        opt_dict['popsize'] = pop_size

    np.warnings.filterwarnings('default')

    self.fdict['cmaes_mode_x'] = x_max_scaled
    self.fdict['cmaes_mode_f'] = f_max
    self.fdict['cmaes_history'] = f_hist, x_hist, seeds
    self.fdict['cmaes_dict'] = opt_dict

    if 'mode_f' in self.fdict.keys() and f_max < self.fdict['mode_f']:
        if done:
            print('[swarms:]'.ljust(
                15, ' ') + " New mode of %s is below old mode of %s. Rejecting..." % (f_max, self.fdict['mode_f']))
    else:
        self.fdict['mode_x'] = x_max_scaled
        self.fdict['mode_f'] = f_max

    if not debug:
        pool.close()

    return f_max, x_max_scaled


def cmaes(self, p0=None, sigma=None, pop_size=None, restart_factor=2, seeds=3, seed=None, linear=None, lprob_seed=None, update_freq=1000, verbose=True, **args):
    """Find mode using CMA-ES from grgrlib.

    Parameters
    ----------
    pop_size : int
        Size of each population. (Default: number of dimensions)
    seeds : in, optional
        Number of different seeds tried. (Default: 3)
    """

    from grgrlib.optimize import cmaes as fmin
    from grgrlib.core import serializer

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
                   verbose=verbose, mapper=self.mapper, **args)

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
