#!/bin/python
# -*- coding: utf-8 -*-

"""contains functions related to (re)compiling the model with different parameters
"""


import numpy as np
import time
from .stats import post_mean


def posterior_sampler(self, nsamples, seed=0, verbose=True):
    """Draw parameters from the posterior.

    Parameters
    ----------
    nsamples : int
        Size of the sample

    Returns
    -------
    array
        Numpy array of parameters
    """
    import random
    random.seed(seed)
    sample = self.get_chain()[-self.get_tune:]
    sample = sample.reshape(-1, sample.shape[(-1)])
    sample = random.choices(sample, k=nsamples)
    return sample


def sample_box(self, dim0, dim1=None, bounds=None, lp_rule=None, verbose=False):
    """Sample from a hypercube
    """
    import chaospy
    bnd = bounds or np.array(self.fdict['prior_bounds'])
    dim1 = dim1 or self.ndim
    rule = lp_rule or 'S'
    res = chaospy.Uniform(0, 1).sample(size=(dim0, dim1), rule=rule)
    res = (bnd[1] - bnd[0]) * res + bnd[0]
    return res


def prior_sampler(self, nsamples, seed=0, test_lprob=False, lks=None, verbose=True, debug=False, **args):
    """Draw parameters from prior. 
    Parameters
    ----------
    nsamples : int
        Size of the prior sample
    seed : int, optional
        Set the random seed (0 by default)
    test_lprob : bool, optional
        Whether to ensure that drawn parameters have a finite likelihood (False by default)
    verbose : bool, optional
    debug : bool, optional
    Returns
    -------
    array
        Numpy array of parameters
    """

    import tqdm
    from grgrlib.core import map2arr
    from grgrlib.multiprocessing import serializer

    l_max, k_max = lks or (None, None)

    if test_lprob and not hasattr(self, 'ndim'):
        self.prep_estim(load_R=True, verbose=verbose > 2)

    frozen_prior = self.fdict.get('frozen_prior')

    if not np.any(frozen_prior):
        from .stats import get_prior
        frozen_prior = get_prior(self.prior, verbose=verbose)[0]

    self.debug |= debug

    if hasattr(self, 'pool'):
        from .estimation import create_pool
        create_pool(self)

    set_par = serializer(self.set_par)
    get_par = serializer(self.get_par)
    lprob = serializer(self.lprob) if test_lprob else None

    def runner(locseed):

        np.random.seed(seed+locseed)
        done = False
        no = 0

        while not done:

            no += 1

            with np.warnings.catch_warnings(record=False):
                try:
                    np.warnings.filterwarnings('error')
                    rst = np.random.randint(2**31)  # win explodes with 2**32
                    pdraw = [pl.rvs(random_state=rst+sn)
                             for sn, pl in enumerate(frozen_prior)]

                    if test_lprob:
                        draw_prob = lprob(pdraw, linear=None,
                                          verbose=verbose > 1)
                        done = not np.isinf(draw_prob)
                    else:
                        set_par(pdraw)
                        done = True

                except Exception as e:
                    if verbose > 1:
                        print(str(e)+' (%s) ' % no)

        return pdraw, no

    if verbose > 1:
        print('[prior_sample:]'.ljust(15, ' ') + ' Sampling from the pior...')

    wrapper = tqdm.tqdm if verbose < 2 else (lambda x, **kwarg: x)
    pmap_sim = wrapper(self.mapper(runner, range(nsamples)), total=nsamples)

    draws, nos = map2arr(pmap_sim)

    if verbose:
        smess = ''
        if test_lprob:
            smess = 'of zero likelihood, '
        print('[prior_sample:]'.ljust(
            15, ' ') + ' Sampling done. %2.2f%% of the prior is either %sindetermined or explosive.' % (100*(sum(nos)-nsamples)/sum(nos), smess))

    return draws


def get_par(self, dummy=None, npar=None, asdict=False, full=True, nsamples=1, verbose=False, roundto=5, debug=False, **args):
    """Get parameters. Tries to figure out what you want. 

    Parameters
    ----------
    dummy : str, optional
        Can be `None`, a parameter name, a parameter set out of {'calib', 'init', 'prior_mean', 'best', 'mode', 'mcmc_mode', 'post_mean', 'posterior_mean'} or one of {'prior', 'post', 'posterior'}. 

        If `None`, returns the current parameters (default). If there are no current parameters, this defaults to 'best'.
        'calib' will return the calibration in the main body of the *.yaml (`parameters`). 
        'init' are the initial values (first column) in the `prior` section of the *.yaml.
        'mode' is the highest known mode from any sort of parameter estimation.
        'best' will default to 'mode' if it exists and otherwise fall back to 'init'.
        'posterior_mean' and 'post_mean' are the same thing.
        'posterior_mode', 'post_mode' and 'mcmc_mode' are the same thing.
        'prior' or 'post'/'posterior' will draw random samples. Obviously, 'posterior', 'mode' etc are only available if a posterior/chain exists.

        NOTE: calling get_par with a set of parameters is the only way to recover the calibrated parameters that are not included in the prior (if you have changed them). All other options will work incrementially on (potential) previous edits of these parameters.

    asdict : bool, optional
        Returns a dict of the values if `True` and an array otherwise (default is `False`).
    full : bool, optional
        Whether to return all parameters or the estimated ones only. (default: True)
    nsamples : int, optional
        Size of the sample. Defaults to 1
    verbose : bool, optional
        Print additional output infmormation (default is `False`)
    roundto : int, optional
        Rounding of additional output if verbose, defaults to 5
    args : various, optional
        Auxilliary arguments passed to `gen_sys` calls

    Returns
    -------
    array or dict
        Numpy array of parameters or dict of parameters
    """
    from .gensys import gen_sys_from_yaml as gen_sys

    if not hasattr(self, 'par'):
        gen_sys(self, verbose=verbose, **args)
    pfnames, pffunc = self.parafunc
    pars_str = [str(p) for p in self.parameters]
    pars = np.array(self.par) if hasattr(
        self, 'par') else np.array(self.par_fix)
    if npar is not None:
        if len(npar) != len(self.par_fix):
            pars[self.prior_arg] = npar
        else:
            pars = npar
    if dummy is None:
        try:
            par_cand = np.array(pars)[self.prior_arg]
        except:
            par_cand = get_par(self, 'best', asdict=False, full=False,
                               verbose=verbose, **args)

    elif not isinstance(dummy, str) and len(dummy) == len(self.par_fix):
        par_cand = dummy[self.prior_arg]
    elif not isinstance(dummy, str) and len(dummy) == len(self.prior_arg):
        par_cand = dummy
    else:
        if dummy in pars_str:
            p = pars[pars_str.index(dummy)]
            if verbose:
                print('[get_par:]'.ljust(15, ' ') + '%s = %s' % (dummy, p))
            return p
        if dummy in pfnames:
            p = pffunc(pars)[pfnames.index(dummy)]
            if verbose:
                print('[get_par:]'.ljust(15, ' ') + '%s = %s' % (dummy, p))
            return p
        if dummy == 'cov_mat':
            gen_sys(self, pars)
            p = self.QQ(self.ppar)
            if verbose:
                print('[get_par:]'.ljust(15, ' ') + '%s = %s' % (dummy, p))
            return p
        if dummy == 'best':
            try:
                par_cand = get_par(self, 'mode', asdict=False, full=False,
                                   verbose=verbose, **args)
            except:
                par_cand = get_par(self, 'init', asdict=False, full=False,
                                   verbose=verbose, **args)

        else:
            old_par = self.par
            pars = self.par_fix
            self.par = self.par_fix
            if dummy == 'prior':
                par_cand = prior_sampler(self, nsamples=nsamples,
                                         verbose=verbose, debug=debug, **args)
            elif dummy in ('post', 'posterior'):
                par_cand = posterior_sampler(self, nsamples=nsamples,
                                             verbose=verbose, **args)
            elif dummy == 'posterior_mean' or dummy == 'post_mean':
                par_cand = post_mean(self)
            elif dummy == 'mode':
                par_cand = self.fdict['mode_x']
            elif dummy in ('mcmc_mode', 'mode_mcmc', 'posterior_mode', 'post_mode'):
                par_cand = self.fdict['mcmc_mode_x']
            elif dummy == 'calib':
                par_cand = self.par_fix[self.prior_arg].copy()
            elif dummy == 'prior_mean':
                par_cand = []
                for pp in self.prior.keys():
                    if self.prior[pp][3] == 'uniform':
                        par_cand.append(
                            0.5 * self.prior[pp][(-2)] + 0.5 * self.prior[pp][(-1)])
                    else:
                        par_cand.append(self.prior[pp][(-2)])

            elif dummy == 'adj_prior_mean':
                par_cand = []
                for pp in self.prior.keys():
                    if self.prior[pp][3] == 'inv_gamma_dynare':
                        par_cand.append(self.prior[pp][(-2)] * 10)
                    else:
                        if self.prior[pp][3] == 'uniform':
                            par_cand.append(
                                0.5 * self.prior[pp][(-2)] + 0.5 * self.prior[pp][(-1)])
                        else:
                            par_cand.append(self.prior[pp][(-2)])

            elif dummy == 'init':
                par_cand = self.fdict['init_value']
                for i in range(self.ndim):
                    if par_cand[i] is None:
                        par_cand[i] = self.par_fix[self.prior_arg][i]

            else:
                self.par = old_par
                raise KeyError(
                    "Parameter or parametrization '%s' does not fit/exist." % dummy)
    if full:
        if isinstance(dummy, str) and dummy in ('prior', 'post', 'posterior'):
            par = np.tile(pars, (nsamples, 1))
            for i in range(nsamples):
                par[i][self.prior_arg] = par_cand[i]

        else:
            par = np.array(pars)
            par[self.prior_arg] = par_cand
        if not asdict:
            return par
        pdict = dict(zip(pars_str, np.round(par, roundto)))
        pfdict = dict(zip(pfnames, np.round(pffunc(par), roundto)))
        return (
            pdict, pfdict)
    if asdict:
        return dict(zip(np.array(pars_str)[self.prior_arg], np.round(par_cand, roundto)))
    if nsamples > 1:
        if dummy not in ('prior', 'post', 'posterior'):
            par_cand = par_cand * \
                (1 + 0.001 * np.random.randn(nsamples, len(par_cand)))
    return par_cand


def get_cov(self, npar=None, **args):
    """get the covariance matrix"""
    return get_par(self, dummy='cov_mat', npar=npar, **args)


def set_par(self, dummy=None, setpar=None, npar=None, verbose=False, return_vv=False, roundto=5, **args):
    """Set the current parameter values.

    In essence, this is a wrapper around `get_par` which also compiles the transition function with the desired parameters.

    Parameters
    ----------
    dummy : str or array, optional
        If an array, sets all parameters. If a string and a parameter name,`setpar` must be provided to define the value of this parameter. Otherwise, `dummy` is forwarded to `get_par` and the returning value(s) are set as parameters.
    setpar : float, optional
        Parametervalue to be set. Of course, only if `dummy` is a parameter name.
    npar : array, optional
        Vector of parameters. If given, this vector will be altered and returnd without recompiling the model. THIS WILL ALTER THE PARAMTER WITHOUT MAKING A COPY!
    verbose : bool
        Whether to output more or less informative messages (defaults to False)
    roundto : int
        Define output precision if output is verbose. (default: 5)
    args : keyword args
        Keyword arguments forwarded to the `gen_sys` call.
    """
    from .gensys import gen_sys_from_yaml as gen_sys

    pfnames, pffunc = self.parafunc
    pars_str = [str(p) for p in self.parameters]
    par = np.array(self.par) if hasattr(
        self, 'par') else np.array(self.par_fix)

    if setpar is None:
        if dummy is None:
            par = get_par(self, dummy=dummy, asdict=False, full=True,
                          verbose=verbose, **args)
        elif len(dummy) == len(self.par_fix):
            par = dummy
        elif len(dummy) == len(self.prior_arg):
            par[self.prior_arg] = dummy
        else:
            par = get_par(self, dummy=dummy, asdict=False, full=True,
                          verbose=verbose, **args)
    elif dummy in pars_str:
        if npar is not None:
            npar = npar.copy()
            if len(npar) == len(self.prior_arg):
                npar[self.prior_names.index(dummy)] = setpar
            else:
                npar[pars_str.index(dummy)] = setpar
            if return_vv:
                return npar, self.vv
            return npar
        par[pars_str.index(dummy)] = setpar
    elif dummy in pfnames:
        raise SyntaxError(
            "Can not set parameter '%s' that is a function of other parameters." % dummy)
    else:
        raise SyntaxError(
            "Parameter '%s' is not defined for this model." % dummy)

    gen_sys(self, par=list(par), verbose=verbose, **args)

    if hasattr(self, 'filter'):
        Q = self.QQ(self.ppar) @ self.QQ(self.ppar)
        self.filter.Q = Q

    if verbose > 1:
        pdict = dict(zip(pars_str, np.round(self.par, roundto)))
        pfdict = dict(zip(pfnames, np.round(pffunc(self.par), roundto)))
        print('[set_par:]'.ljust(15, ' ') +
              ' Parameter(s):\n%s\n%s' % (pdict, pfdict))

    if return_vv:
        return get_par(self), self.vv
    return get_par(self)


def box_check(self, par=None):
    """Check if parameterset lies outside the box constraints

    Parameters
    ----------
    par : array or list, optional
        The parameter set to check
    """

    if par is None:
        par = self.par

    for i, name in enumerate(self.fdict['prior_names']):

        lb, ub = self.fdict['prior_bounds']

        if par[i] < lb[i]:
            print('[box_check:]'.ljust(
                15, ' ') + ' Parameter %s of %s lower than lb of %s.' % (name, par[i].round(5), lb[i]))

        if par[i] > ub[i]:
            print('[box_check:]'.ljust(
                15, ' ') + ' Parameter %s of %s higher than ub of %s.' % (name, par[i].round(5), ub[i]))

    return
