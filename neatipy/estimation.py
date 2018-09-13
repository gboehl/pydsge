#!/bin/python2
# -*- coding: utf-8 -*-
import numpy as np
import warnings
import time

class InvGamma(object):
    
    name = 'inv_gamma'

    def __init__(self, a, b):

        self.a = a
        self.b = b

    def logpdf(self, x):

        from scipy.special import gammaln

        a = self.a
        b = self.b
        
        lpdf    = np.copy(x)

        lpdf[x < 0]     = -1000000000000

        lpdf[x >= 0]    = (np.log(2) - gammaln(b/2) + b/2*np.log(b*a**2/2)
                -(b+1)/2*np.log(x[ x>=0 ]**2) - b*a**2/(2*x**2))

        return lpdf

def wrap_sampler(p0, nwalkers, ndim, ndraws, ncores, info):
    ## very very dirty hack 

    import tqdm
    import pathos
    import emcee

    ## globals are *evil*
    global lprob_global

    ## import the global function and hack it to pretend it is defined on the top level
    def lprob_local(par):
        return lprob_global(par)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lprob_local, pool=pathos.pools.ProcessPool(ncores))

    if not info: np.warnings.filterwarnings('ignore')

    pbar    = tqdm.tqdm(total=ndraws, unit='sample(s)', dynamic_ncols=True)
    for result in sampler.sample(p0, iterations=ndraws):
        pbar.update(1)

    if not info: np.warnings.filterwarnings('default')

    pbar.close()

    return sampler

def bayesian_estimation(self, alpha = 0.2, scale_obs = 0.15, ndraws = 500, tune = 50, ncores = None, nwalkers = 100, find_x0 = True, maxfev = 2500, info = False):

    import pathos
    import scipy.stats as ss
    import scipy.optimize as so
    import tqdm
    from .stats import summary
    from .stats import mc_mean
    from .plots import traceplot, posteriorplot

    if ncores is None:
        ncores    = pathos.multiprocessing.cpu_count()

    self.preprocess(info=info)

    ## dry run before the fun beginns
    self.create_filter(scale_obs = scale_obs)
    self.ukf.R[-1,-1]  /= 100
    self.run_filter()
    print("Model operational. Ready for estimation.")

    par_fix     = np.array(self.par).copy()

    p_names     = [ p.name for p in self.parameters ]
    priors      = self['__data__']['estimation']['prior']
    prior_arg   = [ p_names.index(pp) for pp in priors.keys() ]

    init_par    = par_fix[prior_arg]

    ndim        = len(priors.keys())

    priors_lst     = []
    for pp in priors:
        dist    = priors[str(pp)]
        pmean = dist[1]
        pstdd = dist[2]

        ## simply make use of frozen distributions
        if str(dist[0]) == 'uniform':
            priors_lst.append( ss.uniform(loc=pmean, scale=pmean+pstdd) )
        elif str(dist[0]) == 'inv_gamma':
            priors_lst.append( InvGamma(a=pmean, b=pstdd) )
        elif str(dist[0]) == 'normal':
            priors_lst.append( ss.norm(loc=pmean, scale=pstdd) )
        elif str(dist[0]) == 'gamma':
            priors_lst.append( ss.gamma(loc=pmean, scale=pstdd) )
        elif str(dist[0]) == 'beta':
            a = (1-pmean)*pmean**2/pstdd**2 - pmean
            b = a*(1/pmean - 1)
            priors_lst.append( ss.beta(a=1, b=1) )
        else:
            print('Distribution not implemented')
        print('Adding parameter %s as %s to the prior distributions.' %(pp, dist[0]))


    def llike(parameters):

        if info == 2:
            st  = time.time()

        with warnings.catch_warnings(record=True):
            try: 
                par_fix[prior_arg]  = parameters
                par_active_lst  = list(par_fix)

                self.get_sys(par_active_lst)
                self.preprocess(info=info)

                self.create_filter(scale_obs = scale_obs)
                self.ukf.R[-1,-1]  /= 100
                self.run_filter(info=info)

                if info == 2:
                    print('Sample took '+str(np.round(time.time() - st))+'s.')

                return self.ll

            except:

                if info == 2:
                    print('Sample took '+str(np.round(time.time() - st))+'s. (failure)')

                return -np.inf

    def lprior(pars):
        prior = 0
        for i in range(len(priors_lst)):
            prior   += priors_lst[i].logpdf(pars[i])
        return prior

    def lprob(pars):
        return lprior(pars) + llike(pars)
    
    global lprob_global

    lprob_global    = lprob

    class func_wrap(object):
        ## thats a wrapper to have a progress par in the posterior maximization
        
        name = 'func_wrap'

        def __init__(self, init_par):

            self.n      = 0
            self.maxfev = maxfev
            self.pbar   = tqdm.tqdm(total=maxfev, dynamic_ncols=True)
            self.init_par   = init_par
            self.st     = 0
            self.update_ival    = 1
            self.timer  = 0

        def __call__(self, pars):

            self.res   = -lprob(pars)
            self.x     = pars

            self.n      += 1
            self.timer  += 1

            if self.timer == self.update_ival:
                self.pbar.update(self.update_ival)
                difft   = time.time() - self.st
                if difft < 0.5:
                    self.update_ival *= 2
                if difft > 1 and self.update_ival > 1:
                    self.update_ival /= 2
                self.pbar.set_description('ll: '+str(-self.res.round(5)).rjust(12, ' '))
                self.st  = time.time()
                self.timer  = 0

            if self.n >= maxfev:
                raise StopIteration

            return self.res

        def go(self):

            try:
                res         = so.minimize(self, self.init_par, method='Powell', tol=1e-5)
                self.x      = res['x']
                self.pbar.close()
                print('')
                print(res['message'])
                print('')

            except (KeyboardInterrupt, StopIteration) as e:
                self.pbar.close()
                print('')
                print('Maximum number of function calls exceeded, exiting...')
                print('')

            return self.x

    if find_x0:
        if not info:
            np.warnings.filterwarnings('ignore')
            print('Maximizing posterior distribution... (meanwhile warnings are disabled)')
        else:
            print('Maximizing posterior distribution...')
        result      = func_wrap(init_par).go()
        np.warnings.filterwarnings('default')
        init_par    = result

    print('Initial values:', init_par.round(3))

    pos             = [init_par*(1+1e-2*np.random.randn(ndim)) for i in range(nwalkers)]
    sampler         = wrap_sampler(pos, nwalkers, ndim, ndraws, ncores, info)

    sampler.summary     = lambda: summary(sampler.chain[tune:], priors)
    sampler.traceplot   = lambda **args: traceplot(sampler.chain[tune:], varnames=priors, priors=priors_lst, **args)
    sampler.posteriorplot   = lambda **args: posteriorplot(sampler.chain[tune:], varnames=priors, **args)
    sampler.prior_dist  = priors_lst
    sampler.prior_names = [ pp for pp in priors.keys() ]
    sampler.tune        = tune
    par_mean            = par_fix
    par_mean[prior_arg] = mc_mean(sampler.chain[tune:], varnames=priors)
    sampler.par_means   = list(par_mean)

    self.sampler        = sampler

