#!/bin/python
# -*- coding: utf-8 -*-

from grgrlib import fast0, eig, re_bc
import numpy as np
import numpy.linalg as nl
import time

try:
    from numpy.core._exceptions import UFuncTypeError as ParafuncError
except ModuleNotFoundError:
    ParafuncError = Exception


def get_sys(self, par=None, reduce_sys=None, verbose=False):

    st = time.time()

    if reduce_sys is None:
        try:
            reduce_sys = self.fdict['reduce_sys']
        except KeyError:
            reduce_sys = False

    self.fdict['reduce_sys'] = reduce_sys

    if par is None:
        par = self.p0()

    if not self.const_var:
        raise NotImplementedError('Pakage is only meant to work with OBCs')

    vv_v = np.array([v.name for v in self.variables])
    vv_x = np.array(self.variables)

    dim_v = len(vv_v)

    # obtain matrices from pydsge
    # this could be further accelerated by getting them directly from the equations in pydsge
    AA = self.AA(par)              # forward
    BB = self.BB(par)              # contemp
    CC = self.CC(par)              # backward
    bb = self.bb(par).flatten()    # constraint

    # the special case in which the constraint is just a cut-off of another variable requires
    b = bb.astype(float)

    # define transition shocks -> state
    D = self.PSI(par)

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

    if sum(eig(A).round(3) >= 1) - len(vv_x3):
        raise ValueError('BC *not* satisfied.')

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
    out_msk[-len(vv_v):] = out_msk[-len(vv_v):] & fast0(self.ZZ(par), 0)
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

    self.hx = self.ZZ(par)[:, ~s_out_msk], self.DD(par).squeeze()
    self.obs_arg = np.where(self.hx[0])[1]

    N2 = N[~out_msk][:, ~out_msk]
    A2 = A[~out_msk][:, ~out_msk]
    J2 = J[:, ~out_msk]

    self.SIG = (BB.T @ D)[~s_out_msk]

    self.sys = N2, A2, J2, cx[~out_msk], b2[~out_msk], x_bar

    if verbose:
        print('[get_sys:]'.ljust(15, ' ')+'Creation of system matrices finished in %ss.'
              % np.round(time.time() - st, 3))

    return


def prior_draw(self, nsample, seed=None, ncores=None, verbose=False):
    """Draw parameters from prior. Drawn parameters have a finite likelihood.

    Parameters
    ----------
    nsample : int
        Size of the prior sample
    ncores : int
        Number of cores used for prior sampling. Defaults to the number of available processors

    Returns
    -------
    array
        Numpy array of parameters
    """

    import pathos
    import warnings
    import tqdm
    import cloudpickle as cpickle
    from grgrlib import map2arr

    if seed is None:
        seed = 0

    if verbose:
        print('[get_par:]'.ljust(15, ' ') +
              'Drawing from the pior and checking likelihood.')

    if not hasattr(self, 'ndim'):
        self.prep_estim(load_R=True, verbose=verbose)

    lprob_dump = cpickle.dums(self.lprob)
    lprob = cpickle.loads(lprob_dump)
    frozen_prior = self.fdict['frozen_prior']

    def runner(locseed):

        np.random.seed(seed+locseed)
        draw_prob = -np.inf

        while np.isinf(draw_prob):
            with warnings.catch_warnings(record=False):
                try:
                    np.warnings.filterwarnings('error')
                    rst = np.random.randint(2**16)
                    pdraw = [pl.rvs(random_state=rst)
                             for pl in frozen_prior]
                    draw_prob = lprob(pdraw, None, verbose)
                except:
                    pass

        return pdraw

    if ncores is None:
        ncores = pathos.multiprocessing.cpu_count()

    mapper = map
    if ncores > 1:
        loc_pool = pathos.pools.ProcessPool(ncores)
        loc_pool.clear()
        mapper = loc_pool.imap

    print('[get_cand:]'.ljust(15, ' ') + 'Sampling parameters from prior...')
    pmap_sim = tqdm.tqdm(mapper(
        runner, range(nsample)), total=nsample)

    ## to circumvent mem overflow
    if ncores > 1:
        loc_pool.close()
        loc_pool.join()

    return map2arr(pmap_sim)


def get_par(self, dummy=None, parname=None, asdict=True, full=True, roundto=5, nsample=1, seed=None, ncores=None, verbose=False):
    """Get parameters. Tries to figure out what you want. 

    Parameters
    ----------
    dummy : str, optional
        Can be one of {'mode', 'calib', 'prior_mean', 'init'} or a parameter name. 
    parname : str, optional
        Parameter name if you want to query a single parameter.
    asdict : bool, optional
        Returns a dict of the values if `True` (default) and an array otherwise.
    full : bool, optional
        Whether to return all parameters or the estimated ones only. (default: True)
    nsample : int, optional
        Size of the prior sample
    ncores : int, optional
        Number of cores used for prior sampling. Defaults to the number of available processors

    Returns
    -------
    array or dict
        Numpy array of parameters or dict of parameters
    """

    if not hasattr(self, 'par'):
        get_sys(self, verbose=verbose)

    pfnames, pffunc = self.parafunc
    pars_str = [str(p) for p in self.parameters]

    if parname is None:
        # all with len(par_cand) = len(prior_arg)
        if dummy is None and not asdict:
            dummy = 'mode' if 'mode_x' in self.fdict.keys() else 'init'

        if dummy is None and asdict:
            par_cand = np.array(self.par)[self.prior_arg]
        elif dummy is 'prior':
            return prior_draw(self, nsample, seed, ncores, verbose)
        elif dummy is 'mode':
            par_cand = self.fdict['mode_x']
        elif dummy is 'calib':
            par_cand = self.par_fix[self.prior_arg]
        elif dummy is 'prior_mean':
            par_cand = [self.prior[pp][1] for pp in self.prior.keys()]
        elif dummy is 'init':
            par_cand = self.fdict['init_value']
            for i in range(self.ndim):
                if par_cand[i] is None:
                    par_cand[i] = self.par_fix[self.prior_arg][i]
        else:
            parname = dummy

    if parname is not None:
        if parname in pars_str:
            return self.par[pars_str.index(parname)]
        elif parname in pfnames:
            return pffunc(self.par)[pfnames.index(parname)]
        elif dummy is not None:
            raise KeyError(
                "`which` must be in {'prior', 'mode', 'calib', 'prior_mean', 'init'")
        else:
            raise KeyError("Parameter '%s' does not exist." % parname)

    if asdict and full:
        par = self.par_fix.copy()
        par[self.prior_arg] = par_cand

        pdict = dict(zip(pars_str, np.round(par, roundto)))
        pfdict = dict(zip(pfnames, np.round(pffunc(par), roundto)))

        return pdict, pfdict

    if asdict and not full:
        return dict(zip(pars_str, np.round(par_cand, roundto)))

    if nsample > 1:
        par_cand = par_cand*(1 + 1e-3*np.random.randn(nsample, self.ndim))

    return par_cand


def set_par(self, dummy, setpar=None, roundto=5, autocompile=True, verbose=False):
    """Set the current parameter values.

    Parameter
    ---------
    dummy : str or array
        If an array, sets all parameters. If a string, `setpar` must be provided to define a value.
    setpar : float
        Parametervalue.
    roundto : int
        Define output precision. (default: 5)
    autocompile : bool
        If true, already defines the system and prprocesses matrics. (default: True)
    """

    # if not hasattr(self, 'par'):
    # get_sys(self, verbose=verbose)

    pfnames, pffunc = self.parafunc
    pars_str = [str(p) for p in self.parameters]

    if setpar is None:
        if len(dummy) == len(self.par_fix):
            par = dummy
        elif len(dummy) == len(self.prior_arg):
            par = self.par_fix.copy()
            par[self.prior_arg] = dummy
        else:
            par = self.par_fix.copy()
            par[self.prior_arg] = get_par(
                self, dummy=dummy, parname=None, asdict=False, verbose=verbose)

    elif dummy in pars_str:
        par = self.par_fix.copy()
        par[pars_str.index(dummy)] = setpar
    elif parname in pfnames:
        raise SyntaxError(
            "Can not set parameter '%s' that is function of other parameters." % parname)
    else:
        raise SyntaxError("Parameter '%s' does not exist." % parname)

    get_sys(self, par=list(par), verbose=verbose)
    self.preprocess(verbose=verbose)

    if verbose:
        pdict = dict(zip(pars_str, np.round(self.par, roundto)))
        pfdict = dict(zip(pfnames, np.round(pffunc(self.par), roundto)))

        print('[set_ar:]'.ljust(15, ' ') +
              "Parameter(s):\n%s\n%s" % (pdict, pfdict))

    return
