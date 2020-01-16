#!/bin/python
# -*- coding: utf-8 -*-

"""contains functions related to (re)compiling the model with different parameters
"""

from grgrlib import fast0, eig, re_bc
import numpy as np
import numpy.linalg as nl
import time
from .engine import preprocess
from .stats import post_mean

try:
    from numpy.core._exceptions import UFuncTypeError as ParafuncError
except ModuleNotFoundError:
    ParafuncError = Exception


def get_sys(self, par=None, reduce_sys=None, l_max=None, k_max=None, verbose=False):
    """Creates the transition function given a set of parameters. 

    If no parameters are given this will default to the calibration in the `yaml` file.

    Parameters
    ----------
    par : array or list, optional
        The parameters to parse into the transition function. (defaults to calibration in `yaml`)
    reduce_sys : bool, optional
        If true, the state space is reduced. This speeds up computation.
    l_max : int, optional
        The expected number of periods *until* the constraint binds (defaults to 3).
    k_max : int, optional
        The expected number of periods for which the constraint binds (defaults to 17).
    """

    st = time.time()

    if reduce_sys is None:
        try:
            reduce_sys = self.fdict['reduce_sys']
        except KeyError:
            reduce_sys = False

    l_max = 3 if l_max is None else l_max
    k_max = 17 if k_max is None else k_max

    self.fdict['reduce_sys'] = reduce_sys

    par = self.p0() if par is None else list(par)
    ppar = self.compile(par)  # parsed par

    if not self.const_var:
        raise NotImplementedError('Pakage is only meant to work with OBCs')

    vv_v = np.array([v.name for v in self.variables])
    vv_x = np.array(self.variables)

    dim_v = len(vv_v)

    # obtain matrices from pydsge
    # this could be further accelerated by getting them directly from the equations in pydsge
    AA = self.AA(ppar)              # forward
    BB = self.BB(ppar)              # contemp
    CC = self.CC(ppar)              # backward
    bb = self.bb(ppar).flatten()    # constraint

    # the special case in which the constraint is just a cut-off of another variable requires
    b = bb.astype(float)

    # define transition shocks -> state
    D = self.PSI(ppar)

    # mask those vars that are either forward looking or part of the constraint
    in_x = ~fast0(AA, 0) | ~fast0(b[:dim_v])

    # reduce x vector
    vv_x2 = vv_x[in_x]
    A1 = AA[:, in_x]
    b1 = np.hstack((b[:dim_v][in_x], b[dim_v:]))

    dim_x = len(vv_x2)

    # define actual matrices
    M = np.block([[np.zeros(A1.shape), CC],
                  [np.eye(dim_x), np.zeros((dim_x, dim_v))]])

    P = np.block([[A1, -BB],
                  [np.zeros((dim_x, dim_x)), np.eye(dim_v)[in_x]]])

    c_arg = list(vv_x2).index(self.const_var)

    # c contains information on how the constraint var affects the system
    c_M = M[:, c_arg]
    c_P = P[:, c_arg]

    # get rid of constrained var
    b2 = np.delete(b1, c_arg)
    M1 = np.delete(M, c_arg, 1)
    P1 = np.delete(P, c_arg, 1)
    vv_x3 = np.delete(vv_x2, c_arg)

    # decompose P in singular & nonsingular rows
    U, s, V = nl.svd(P1)
    s0 = fast0(s)

    P2 = np.diag(s) @ V
    M2 = U.T @ M1

    c1 = U.T @ c_M

    if not fast0(c1[s0], 2) or not fast0(U.T[s0] @ c_P, 2):
        NotImplementedError(
            'The system depends directly or indirectly on whether the constraint holds in the future or not.\n')

    # actual desingularization by iterating equations in M forward
    P2[s0] = M2[s0]

    if 'x_bar' in [p.name for p in self.parameters]:
        x_bar = par[[p.name for p in self.parameters].index('x_bar')]
    elif 'x_bar' in self.parafunc[0]:
        pf = self.parafunc
        x_bar = pf[1](par)[pf[0].index('x_bar')]
    else:
        print("Parameter `x_bar` (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        x_bar = -1

    # create the stuff that the algorithm needs
    N = nl.inv(P2) @ M2
    A = nl.inv(P2) @ (M2 + np.outer(c1, b2))

    # rounding here must correspond to the rounding in re_bc
    # if sum(eig(A).round(8) >= 1) != len(vv_x3):
    # if sum(eig(A).round(8) < 1) != len(vv_v):
    # raise ValueError('B-K condition *not* satisfied.')

    dim_x = len(vv_x3)
    OME = re_bc(A, dim_x)
    J = np.hstack((np.eye(dim_x), -OME))

    try:
        cx = nl.inv(P2) @ c1*x_bar
    except ParafuncError:
        raise SyntaxError(
            "At least one parameter should rather be a function of parameters ('parafunc')...")

    # check condition:
    n1 = N[:dim_x, :dim_x]
    n3 = N[dim_x:, :dim_x]
    cc1 = cx[:dim_x]
    cc2 = cx[dim_x:]
    bb1 = b2[:dim_x]

    out_msk = fast0(N, 0) & fast0(A, 0) & fast0(b2) & fast0(cx)
    out_msk[-len(vv_v):] = out_msk[-len(vv_v):] & fast0(self.ZZ(ppar), 0)
    # store those that are/could be reduced
    self.out_msk = out_msk[-len(vv_v):].copy()

    if not reduce_sys:
        out_msk[-len(vv_v):] = False

    s_out_msk = out_msk[-len(vv_v):]

    if hasattr(self, 'P'):
        if self.P.shape[0] < sum(~s_out_msk):
            P_new = np.zeros((len(self.out_msk), len(self.out_msk)))
            if P_new[~self.out_msk][:, ~self.out_msk].shape != self.P.shape:
                print('[get_sys:]'.ljust(
                    15, ' ')+' Shape missmatch of P-matrix, number of states seems to differ!')
            P_new[~self.out_msk][:, ~self.out_msk] = self.P
            self.P = P_new
        elif self.P.shape[0] > sum(~s_out_msk):
            self.P = self.P[~s_out_msk][:, ~s_out_msk]

    # add everything to the DSGE object
    self.vv = vv_v[~s_out_msk]
    self.vx = np.array([v.name for v in vv_x3])
    self.dim_x = dim_x
    self.dim_v = len(self.vv)

    self.par = par
    self.ppar = ppar

    self.hx = self.ZZ(ppar)[:, ~s_out_msk], self.DD(ppar).squeeze()
    self.obs_arg = np.where(self.hx[0])[1]

    N2 = N[~out_msk][:, ~out_msk]
    A2 = A[~out_msk][:, ~out_msk]
    J2 = J[:, ~out_msk]

    self.SIG = (BB.T @ D)[~s_out_msk]

    self.sys = N2, A2, J2, cx[~out_msk], b2[~out_msk], x_bar

    if verbose:
        print('[get_sys:]'.ljust(15, ' ')+'Creation of system matrices finished in %ss.'
              % np.round(time.time() - st, 3))

    preprocess(self, l_max, k_max, verbose)

    test = self.precalc_mat[0][1, 0, 1]
    if (eig(test[-test.shape[1]:]) > 1).any():
        raise ValueError('Explosive dynamics detected.')

    return


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
    from .clsmethods import get_tune

    random.seed(seed)
    sample = self.get_chain()[-get_tune(self):]
    sample = sample.reshape(-1, sample.shape[-1])
    sample = random.choices(sample, k=nsamples)

    return sample


def sample_box(self, dim0, dim1=None, bounds=None, lp_rule=None, verbose=False):
    """Sample from a hypercube
    """

    # TODO: include in get_par

    import chaospy

    bnd = bounds or np.array(self.fdict['prior_bounds'])
    dim1 = dim1 or self.ndim
    rule = lp_rule or 'S'

    res = chaospy.Uniform(0, 1).sample(size=(dim0, dim1), rule=rule)
    res = (bnd[1] - bnd[0])*res + bnd[0]

    return res


def prior_sampler(self, nsamples, seed=0, test_lprob=False, verbose=True):
    """Draw parameters from prior. Drawn parameters have a finite likelihood.

    Parameters
    ----------
    nsamples : int
        Size of the prior sample

    Returns
    -------
    array
        Numpy array of parameters
    """

    import tqdm
    from grgrlib import map2arr, serializer

    reduce_sys = np.copy(self.fdict['reduce_sys'])

    if not reduce_sys:
        self.get_sys(reduce_sys=True, verbose=verbose > 1)

    if test_lprob and not hasattr(self, 'ndim'):
        self.prep_estim(load_R=True, verbose=verbose > 2)

    if not 'frozen_prior' in self.fdict.keys():
        from .stats import get_prior
        self.fdict['frozen_prior'] = get_prior(self.prior)[0]

    frozen_prior = self.fdict['frozen_prior']

    if hasattr(self, 'pool'):
        self.pool.clear()

    get_par = serializer(self.get_par)
    get_sys = serializer(self.get_sys)
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
                    rst = np.random.randint(2**32-2)
                    pdraw = [pl.rvs(random_state=rst+sn)
                             for sn, pl in enumerate(frozen_prior)]

                    if test_lprob:
                        draw_prob = lprob(pdraw, linear=None,
                                          verbose=verbose > 1)
                        done = not np.isinf(draw_prob)
                    else:
                        pdraw_full = get_par(pdraw, asdict=False, full=True)
                        get_sys(par=pdraw_full, reduce_sys=True)
                        done = True

                except Exception as e:
                    if verbose > 1:
                        print(str(e)+'(%s) ' % no)

        return pdraw, no

    if verbose > 1:
        print('[prior_sample:]'.ljust(15, ' ') + ' sampling from the pior...')

    wrapper = tqdm.tqdm if verbose < 2 else (lambda x, **kwarg: x)
    pmap_sim = wrapper(self.mapper(runner, range(nsamples)), total=nsamples)

    draws, nos = map2arr(pmap_sim)

    if not reduce_sys:
        self.get_sys(reduce_sys=False, verbose=verbose > 1)

    if verbose:
        smess = ''
        if test_lprob:
            smess = 'of zero likelihood, '
        print('[prior_sample:]'.ljust(
            15, ' ') + ' sampling done. %2.2f%% of the prior are either %s indetermined or explosive.' % (100*(sum(nos)-nsamples)/nsamples, smess))

    return draws


def get_par(self, dummy=None, parname=None, asdict=False, full=None, nsamples=1, verbose=False, roundto=5, **args):
    """Get parameters. Tries to figure out what you want. 

    Parameters
    ----------
    dummy : str, optional
    Can be `None`, a parameter name, or one of {'calib', 'init', 'prior_mean', 'best', 'mode', 'mcmc_mode', 'post_mean', 'posterior_mean', 'prior', 'posterior'}. 
    If `None`, returns the current parameters (default).
    If a parameter name, this is equivalent to setting `parname`.
    Otherwise, 'calib' will return the calibration in the main body of the *.yaml (`parameters`). 
    'init' are the initial values (first column) in the `prior` section of the *.yaml.
    'posterior_mean' and 'post_mean' are the same thing.
    'prior' or 'posterior' will draw random samples. Obviously, 'posterior', 'mode' etc are only available if a posterior/chain exists.
    parname : str, optional
        Parameter name if you want to query a single parameter.
    asdict : bool, optional
        Returns a dict of the values if `True` and an array otherwise (default is `False`).
    full : bool, optional
        Whether to return all parameters or the estimated ones only. (default: False unless asdict is True)
    nsamples : int, optional
        Size of the sample. Defaults to 1
    verbose : bool, optional
        Print additional output infmormation (default is `False`)
    roundto : int, optional
        Rounding of additional output if verbose, defaults to 5
    args : various, optional
        Auxilliary arguments passed on to a sampler

    Returns
    -------
    array or dict
        Numpy array of parameters or dict of parameters
    """

    full = full if full is not None else asdict

    if not hasattr(self, 'par'):
        get_sys(self, verbose=verbose)

    pfnames, pffunc = self.parafunc
    pars_str = [str(p) for p in self.parameters]

    if parname is None:
        if dummy is None:
            try:
                par_cand = np.array(self.par)[self.prior_arg]
            except:
                par_cand = get_par(self, 'best', asdict=False, full=False)
        elif len(dummy) == len(self.par_fix):
            par_cand = dummy[self.prior_arg]
        elif len(dummy) == len(self.prior_arg):
            par_cand = dummy
        elif dummy == 'best':
            try:
                par_cand = get_par(self, 'mode', asdict=False, full=False)
            except:
                par_cand = get_par(self, 'init', asdict=False, full=False)
        elif dummy == 'prior':
            return prior_sampler(self, nsamples=nsamples, verbose=verbose, **args)
        elif dummy == 'posterior':
            return posterior_sampler(self, nsamples=nsamples, verbose=verbose, **args)
        elif dummy == 'posterior_mean' or dummy == 'post_mean':
            par_cand = post_mean(self)
        elif dummy == 'mode':
            par_cand = self.fdict['mode_x']
        elif dummy == 'mcmc_mode' or dummy == 'mode_mcmc':
            par_cand = self.fdict['mcmc_mode_x']
        elif dummy == 'calib':
            par_cand = self.par_fix[self.prior_arg]
        elif dummy == 'prior_mean':
            par_cand = [self.prior[pp][-2] for pp in self.prior.keys()]
        elif dummy == 'adj_prior_mean':
            # adjust for prior[pp][-2] not beeing the actual mean for inv_gamma_dynare
            par_cand = [self.prior[pp][-2]*10 **
                        (self.prior[pp][3] == 'inv_gamma_dynare') for pp in self.prior.keys()]
        elif dummy == 'init':
            par_cand = self.fdict['init_value']
            for i in range(self.ndim):
                if par_cand[i] is None:
                    par_cand[i] = self.par_fix[self.prior_arg][i]
        else:
            return get_par(self, parname=dummy, full=full)
    elif parname in pars_str:
        return self.par[pars_str.index(parname)]
    elif parname in pfnames:
        return pffunc(self.par)[pfnames.index(parname)]
    elif dummy is not None:
        raise KeyError(
            "`which` must be in {'prior', 'mode', 'calib', 'prior_mean', 'init'")
    else:
        raise KeyError("Parameter '%s' does not exist." % parname)

    if full:
        par = self.par.copy() if hasattr(self, 'par') else self.par_fix.copy()
        par = np.array(par)
        par[self.prior_arg] = par_cand

        if not asdict:
            return par

        pdict = dict(zip(pars_str, np.round(par, roundto)))
        pfdict = dict(zip(pfnames, np.round(pffunc(par), roundto)))

        return pdict, pfdict

    if asdict:
        return dict(zip(np.array(pars_str)[self.prior_arg], np.round(par_cand, roundto)))

    if nsamples > 1:
        par_cand = par_cand*(1 + 1e-3*np.random.randn(nsamples, len(par_cand)))

    return par_cand


def set_par(self, dummy, setpar=None, par=None, autocompile=True, verbose=False, roundto=5, **args):
    """Set the current parameter values.

    In essence, this is a wrapper around `get_par` which also compiles the transition function with the desired parameters.

    Parameters
    ----------
    dummy : str or array
        If an array, sets all parameters. If a string and a parameter name,`setpar` must be provided to define the value of this parameter. Otherwise, `dummy` is forwarded to `get_par` and the returning value(s) are set as parameters.
    setpar : float
        Parametervalue.
    autocompile : bool
        If true, already defines the system and prprocesses matrics. (default: True)
    verbose : bool
        Whether to output more or less informative messages (defaults to False)
    roundto : int
        Define output precision if output is verbose. (default: 5)
    args : keyword args
        Keyword arguments forwarded to the `get_sys` call.
    """

    pfnames, pffunc = self.parafunc
    pars_str = [str(p) for p in self.parameters]

    if setpar is None:
        if len(dummy) == len(self.par_fix):
            par = dummy
        elif len(dummy) == len(self.prior_arg):
            par = np.copy(self.par) if hasattr(
                self, 'par') else np.copy(self.par_fix)
            par[self.prior_arg] = dummy
        else:
            par = get_par(self, dummy=dummy, parname=None,
                          asdict=False, full=True, verbose=verbose)

    elif dummy in pars_str:
        if par is None:
            par = self.par.copy() if hasattr(self, 'par') else self.par_fix.copy()
        elif len(par) == len(self.prior_arg):
            npar = self.par.copy() if hasattr(self, 'par') else self.par_fix.copy()
            npar = np.array(npar)
            npar[self.prior_arg] = par
            par = npar
        par[pars_str.index(dummy)] = setpar
    elif dummy in pfnames:
        raise SyntaxError(
            "Can not set parameter '%s' that is function of other parameters." % parname)
    else:
        raise SyntaxError(
            "Parameter '%s' is not defined for this model." % parname)

    get_sys(self, par=list(par), verbose=verbose, **args)

    if hasattr(self, 'filter'):

        self.filter.eps_cov = self.QQ(self.ppar)

        if self.filter.name == 'KalmanFilter':
            CO = self.SIG @ self.filter.eps_cov
            Q = CO @ CO.T
        elif self.filter.name == 'ParticleFilter':
            raise NotADirectoryError
        else:
            Q = self.QQ(self.ppar) @ self.QQ(self.ppar)

        self.filter.Q = Q

    if verbose:
        pdict = dict(zip(pars_str, np.round(self.par, roundto)))
        pfdict = dict(zip(pfnames, np.round(pffunc(self.par), roundto)))

        print('[set_ar:]'.ljust(15, ' ') +
              "Parameter(s):\n%s\n%s" % (pdict, pfdict))

    return par
