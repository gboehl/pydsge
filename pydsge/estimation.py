#!/bin/python2
# -*- coding: utf-8 -*-
import numpy as np
import warnings
import os
import time
import emcee
from .stats import InvGamma, summary, mc_mean

def mcmc(p0, nwalkers, ndim, ndraws, priors, sampler, ntemp, ncores, update_freq, description, verbose):
    ## very very dirty hack 

    import tqdm
    import pathos

    ## globals are *evil*
    global lprob_global
    global llike_global
    global lprior_global

    ## import the global function and hack it to pretend it is defined on the top level
    def lprob_local(par):
        return lprob_global(par)

    def llike_local(par):
        return llike_global(par)
    
    def lprior_local(par):
        return lprior_global(par)

    loc_pool    = pathos.pools.ProcessPool(ncores)

    if sampler is 'ptes':
        sampler = emcee.PTSampler(ntemps = ntemp, nwalkers = nwalkers, dim = ndim, logp = lprior_local, logl = llike_local, pool = loc_pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lprob_local, pool = loc_pool)


    if not verbose: np.warnings.filterwarnings('ignore')

    pbar    = tqdm.tqdm(total=ndraws, unit='sample(s)', dynamic_ncols=True)

    cnt     = 0
    for result in sampler.sample(p0, iterations=ndraws):
        if update_freq and pbar.n and not pbar.n % update_freq:
            pbar.write('')
            if description is not None:
                pbar.write('[bayesian_estimation -> mcmc:]'.ljust(45, ' ')+'Summary from last %s of %s iterations (%s):' %(update_freq, pbar.n, str(description)))
            else:
                pbar.write('[bayesian_estimation -> mcmc:]'.ljust(45, ' ')+' Summary from last %s of %s iterations:' %(update_freq, pbar.n))
            pbar.write(str(summary(sampler.chain.reshape(-1, ndraws, ndim)[:,pbar.n-update_freq:pbar.n,:], priors).round(3)))
            pbar.write("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        pbar.update(1)

    loc_pool.close()
    loc_pool.join()
    loc_pool.clear()

    if not verbose: np.warnings.filterwarnings('default')

    pbar.close()

    return sampler


def bayesian_estimation(self, N = None, P = None, R = None, ndraws = 500, tune = None, ncores = None, nwalkers = 100, ntemp = 4, maxfev = 2500, pmdm_method = None, sampler = None, update_freq = None, verbose = False):

    import pathos
    import scipy.stats as ss
    import scipy.optimize as so
    import tqdm
    from .plots import traceplot, posteriorplot

    if ncores is None:
        ncores    = pathos.multiprocessing.cpu_count()

    if tune is None:
        tune    = int(ndraws*4/5.)

    if update_freq is None:
        update_freq     = int(ndraws/4.)

    description     = None

    if hasattr(self, 'description'):
        description     = self.description

    if P is None:
        P   = np.eye(len(self.vv))*1e1

    self.preprocess(verbose=verbose)

    ## dry run before the fun beginns
    self.create_filter(P = P, R = R, N = N)
    self.get_ll(verbose = verbose)

    print()
    print('[bayesian_estimation:]'.ljust(30, ' ')+'Model operational. Ready for estimation.')
    print()

    par_fix     = np.array(self.par).copy()

    p_names     = [ p.name for p in self.parameters ]
    priors      = self['__data__']['estimation']['prior']
    prior_arg   = [ p_names.index(pp) for pp in priors.keys() ]

    ## add to class so that it can be stored later
    self.par_fix    = par_fix
    self.prior_arg  = prior_arg
    self.ndraws     = ndraws

    init_par    = par_fix[prior_arg]

    ndim        = len(priors.keys())

    print('[bayesian_estimation:]'.ljust(30, ' ')+'Adding parameters to the prior distribution:')

    priors_lst     = []
    for pp in priors:
        dist    = priors[str(pp)]
        pmean = dist[1]
        pstdd = dist[2]

        ## simply make use of frozen distributions
        if str(dist[0]) == 'uniform':
            priors_lst.append( ss.uniform(loc=pmean, scale=pstdd-pmean) )
        elif str(dist[0]) == 'inv_gamma':
            priors_lst.append( InvGamma(a=pmean, b=pstdd) )
        elif str(dist[0]) == 'normal':
            priors_lst.append( ss.norm(loc=pmean, scale=pstdd) )
        elif str(dist[0]) == 'gamma':
            b = pstdd**2/pmean
            a = pmean/b
            priors_lst.append( ss.gamma(a, scale=b) )
        elif str(dist[0]) == 'beta':
            a = (1-pmean)*pmean**2/pstdd**2 - pmean
            b = a*(1/pmean - 1)
            priors_lst.append( ss.beta(a=1, b=1) )
        else:
            raise ValueError(' Distribution *not* implemented: ', str(dist[0]))
        print('     parameter %s as %s with mean %s and std %s...' %(pp, dist[0], pmean, pstdd))


    def llike(parameters):

        if verbose == 2:
            st  = time.time()

        with warnings.catch_warnings(record=True):
            try: 
                warnings.filterwarnings('error')
                
                par_fix[prior_arg]  = parameters
                par_active_lst  = list(par_fix)

                self.get_sys(par = par_active_lst, reduce_sys = True, verbose = verbose)
                self.preprocess(verbose=verbose)

                self.create_filter(P = P, R = R, N = N)

                ll  = self.get_ll(verbose=verbose)

                if verbose == 2:
                    print('[bayesian_estimation -> llike:]'.ljust(30, ' ')+'Sample took '+str(np.round(time.time() - st, 3))+'s.')

                return ll

            except:

                if verbose == 2:
                    print('[bayesian_estimation -> llike:]'.ljust(30, ' ')+'Sample took '+str(np.round(time.time() - st, 3))+'s. (failure)')

                return -np.inf

    def lprior(pars):

        prior = 0
        for i in range(len(priors_lst)):
            prior   += priors_lst[i].logpdf(pars[i])

        return prior

    def lprob(pars):
        return lprior(pars) + llike(pars)
    
    global lprob_global
    global llike_global
    global lprior_global

    lprob_global    = lprob
    llike_global    = llike
    lprior_global   = lprior
    prior_names     = [ pp for pp in priors.keys() ]

    class pmdm(object):
        ## thats a wrapper to have a progress par in the posterior maximization
        
        name = 'pmdm'

        def __init__(self, init_par):

            self.n      = 0
            self.maxfev = maxfev
            self.pbar   = tqdm.tqdm(total=maxfev, dynamic_ncols=True)
            self.init_par   = init_par
            self.st     = 0
            self.update_ival    = 1
            self.timer  = 0
            self.res_max    = np.inf

        def __call__(self, pars):

            self.res   = -lprob(pars)
            self.x     = pars

            ## better ensure we're not just running with the wolfs when maxfev is hit
            if self.res < self.res_max:
                self.res_max    = self.res
                self.x_max      = self.x

            self.n      += 1
            self.timer  += 1

            if self.timer == self.update_ival:

                self.pbar.update(self.update_ival)

                difft   = time.time() - self.st
                if difft < 1:
                    self.update_ival *= 2
                if difft > 2 and self.update_ival > 1:
                    self.update_ival /= 2

                self.pbar.set_description('ll: '+str(-self.res.round(5)).rjust(12, ' ')+' ['+str(-self.res_max.round(5))+']')
                self.st  = time.time()
                self.timer  = 0

            ## prints information snapshots
            if update_freq and not self.n % update_freq:
                ## getting the number of colums isn't that easy
                with os.popen('stty size', 'r') as rows_cols:
                    cols            = rows_cols.read().split()[1]
                if description is not None:
                    self.pbar.write('[bayesian_estimation -> pmdm:]'.ljust(30, ' ')+'Current best guess @ iteration %s and ll of %s (%s):' %(self.n, self.res_max.round(5), str(description)))
                else:
                    self.pbar.write('[bayesian_estimation -> pmdm:]'.ljust(30, ' ')+'Current best guess @ iteration %s and ll of %s):' %(self.n, self.res_max.round(5)))
                ## split the info such that it is readable
                lnum            = (len(priors)*8)//(int(cols)-8) + 1
                priors_chunks   = np.array_split(np.array(prior_names), lnum)
                vals_chunks     = np.array_split([round(m_val, 3) for m_val in self.x_max], lnum)
                for pchunk, vchunk in zip(priors_chunks, vals_chunks):
                    row_format ="{:>8}" * (len(pchunk) + 1)
                    self.pbar.write(row_format.format("", *pchunk))
                    self.pbar.write(row_format.format("", *vchunk))
                    self.pbar.write('')
                self.pbar.write('')


            if self.n >= maxfev:
                raise StopIteration

            return self.res

        def go(self):

            try:
                f_val       = -np.inf
                self.x      = self.init_par

                res         = so.minimize(self, self.x, method=pmdm_method, tol=1e-1)

                self.pbar.close()
                print('')
                if self.res_max < res['fun']:
                    print('[bayesian_estimation -> pmdm:]', res['message'], 'Maximization returned value lower than actual (known) optimum ('+str(-self.res_max)+' > '+str(-self.res)+').')
                else:
                    print('[bayesian_estimation -> pmdm:]', res['message'], 'Log-likelihood is '+str(np.round(-res['fun'],5))+'.')
                print('')

            except (KeyboardInterrupt, StopIteration) as e:
                self.pbar.close()
                print('')
                print('[bayesian_estimation -> pmdm:] Maximum number of function calls exceeded, exiting. Log-likelihood is '+str(np.round(-self.res_max,5))+'...')
                print('')

            return self.x_max

    if maxfev:

        print()
        if not verbose:
            np.warnings.filterwarnings('ignore')
            print('[bayesian_estimation -> pmdm:] Maximizing posterior mode density (meanwhile warnings are disabled)')
        else:
            print('[bayesian_estimation -> pmdm:] Maximizing posterior mode density:')
        if pmdm_method is None:
            pmdm_method     = 'Powell'
        elif isinstance(pmdm_method, int):
            methodl     = ["L-BFGS-B", "Nelder-Mead", "Powell", "CG", "BFGS", "TNC", "COBYLA"]
            pmdm_method  = methodl[pmdm_method]
            print('[bayesian_estimation -> pmdm:] Using %s for optimization. Available methods are %s.' %(pmdm_method, ', '.join(methodl)))
        print()

        result      = pmdm(init_par).go()
        np.warnings.filterwarnings('default')
        init_par    = result

    print()
    print('[bayesian_estimation:] Inital values for MCMC:')
    with os.popen('stty size', 'r') as rows_cols:
        cols            = rows_cols.read().split()[1]
        lnum            = (len(priors)*8)//(int(cols)-8) + 1
        priors_chunks   = np.array_split(np.array(prior_names), lnum)
        vals_chunks     = np.array_split([round(m_val, 3) for m_val in init_par], lnum)
        for pchunk, vchunk in zip(priors_chunks, vals_chunks):
            row_format ="{:>8}" * (len(pchunk) + 1)
            print(row_format.format("", *pchunk))
            print(row_format.format("", *vchunk))
            print()


    if ndraws:
        print()
        if sampler == 'ptes':
            # pos             = [init_par*(1+1e-3*np.random.randn(ntemp, ndim)) for i in range(nwalkers)]
            pos             = init_par*(1+1e-3*np.random.randn(ntemp, nwalkers, ndim))
        else:
            pos             = [init_par*(1+1e-3*np.random.randn(ndim)) for i in range(nwalkers)]
        sampler         = mcmc(pos, nwalkers, ndim, ndraws, priors, sampler, ntemp, ncores, update_freq, description, verbose)

        print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

        self.chain          = sampler.chain.reshape(-1, ndraws, ndim)

        sampler.summary     = lambda: summary(self.chain[:,tune:,:], priors)
        sampler.traceplot   = lambda **args: traceplot(self.chain, varnames=priors, tune=tune, priors=priors_lst, **args)
        sampler.posteriorplot   = lambda **args: posteriorplot(self.chain, varnames=priors, tune=tune, **args)

        par_mean            = par_fix
        par_mean[prior_arg] = mc_mean(self.chain[:,tune:], varnames=priors)

        sampler.prior_dist  = priors_lst
        sampler.prior_names = prior_names
        sampler.tune        = tune
        sampler.par_means   = list(par_mean)

        self.sampler        = sampler
