#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import warnings
from numba import njit
import time
from grgrlib import *
from .engine import boehlgorithm

def get_sys(self, par=None, reduce_sys = True, verbose = False):

    st  = time.time()

    if par is None:
        par     = self.p0()

    if not self.const_var:
        warnings.warn('Code is only meant to work with OBCs')

    vv_v    = np.array([v.name for v in self.variables])
    vv_x    = np.array(self.variables)

    dim_v   = len(vv_v)

    ## obtain matrices from pydsge
    ## this can be further accelerated by getting them directly from the equations in pydsge
    AA  = self.AA(par)              # forward
    BB  = self.BB(par)              # contemp
    CC  = self.CC(par)              # backward
    bb  = self.bb(par).flatten()    # constraint

    ## the special case in which the constraint is just a cut-off of another variable requires
    b   = bb.astype(float)

    ## define transition shocks -> state
    D   = self.PSI(par)

    ## mask those vars that are either forward looking or part of the constraint
    in_x       = ~fast0(AA, 0) | ~fast0(b[:dim_v])

    ## reduce x vector
    vv_x2   = vv_x[in_x]
    A1      = AA[:,in_x]
    b1      = np.hstack((b[:dim_v][in_x], b[dim_v:]))

    dim_x   = len(vv_x2)

    ## define actual matrices
    M       = np.block([[np.zeros(A1.shape), CC], 
                        [np.eye(dim_x), np.zeros((dim_x,dim_v))]])

    P       = np.block([[A1, -BB],
                        [np.zeros((dim_x,dim_x)), np.eye(dim_v)[in_x]]])
    
    c_arg           = list(vv_x2).index(self.const_var)

    ## c contains information on how the constraint var affects the system
    c_M     = M[:,c_arg]
    c_P     = P[:,c_arg]

    ## get rid of constrained var
    b2      = np.delete(b1, c_arg)
    M1      = np.delete(M, c_arg, 1)
    P1      = np.delete(P, c_arg, 1)
    vv_x3   = np.delete(vv_x2, c_arg)

    ## decompose P in singular & nonsingular rows
    U, s, V     = nl.svd(P1)
    s0  = fast0(s)

    P2  = np.diag(s) @ V
    M2  = U.T @ M1

    c1  = U.T @ c_M

    if not fast0(c1[s0], 2) or not fast0(U.T[s0] @ c_P, 2):
        warnings.warn('\nNot implemented: the system depends directly or indirectly on whether the constraint holds in the future or not.\n')
        
    ## actual desingularization by iterating equations in M forward
    P2[s0]  = M2[s0]


    if 'x_bar' in [p.name for p in self.parameters]:
        x_bar   = par[[p.name for p in self.parameters].index('x_bar')]
    elif 'x_bar' in self.parafunc[0]:
        pf      = self.parafunc
        x_bar   = pf[1](par)[pf[0].index('x_bar')]
    else:
        warnings.warn("\nx_bar (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.\n")
        x_bar       = -1

    ## create the stuff that the algorithm needs
    N       = nl.inv(P2) @ M2 
    A       = nl.inv(P2) @ (M2 + np.outer(c1,b2))

    if sum(eig(A).round(3) >= 1) - len(vv_x3):
        raise ValueError('BC *not* satisfied.')

    dim_x       = len(vv_x3)
    OME         = re_bc(A, dim_x)
    J 			= np.hstack((np.eye(dim_x), -OME))
    cx 		    = nl.inv(P2) @ c1*x_bar

    ## check condition:
    n1  = N[:dim_x,:dim_x]
    n3  = N[dim_x:,:dim_x]
    cc1  = cx[:dim_x]
    cc2  = cx[dim_x:]
    bb1  = b2[:dim_x]

    out_msk     = fast0(N, 0) & fast0(A, 0) & fast0(b2) & fast0(cx)
    out_msk[-len(vv_v):]    = out_msk[-len(vv_v):] & fast0(self.ZZ(par), 0)

    if not reduce_sys:
        out_msk[-len(vv_v):][:]  = False

    ## add everything to the DSGE object
    self.vv     = vv_v[~out_msk[-len(vv_v):]]
    self.vx     = np.array([v.name for v in vv_x3])
    self.dim_x  = dim_x
    self.dim_v  = len(self.vv)

    self.observables    = self['observables']
    self.par    = par

    self.hx             = self.ZZ(par)[:,~out_msk[-len(vv_v):]], self.DD(par).squeeze()
    self.obs_arg        = np.where(self.hx[0])[1]

    N2  = N[~out_msk][:,~out_msk]
    A2  = A[~out_msk][:,~out_msk]
    J2  = J[:,~out_msk]

    self.SIG    = (BB.T @ D)[~out_msk[-len(vv_v):]]

    self.sys 	= N2, A2, J2, cx[~out_msk], b2[~out_msk], x_bar

    if verbose:
        print('[get_sys:]'.ljust(15, ' ')+'Creation of system matrices finished in %ss.'
              % np.round(time.time() - st,3))



def irfs(self, shocklist, wannasee=None, use_bruite=2, show_warnings=True, verbose=False):

    ## returns time series of impule responses 
    ## shocklist: takes list of tuples of (shock, size, timing) 
    ## wannasee: list of strings of the variables to be plotted and stored

    slabels     = [v.replace('_','') for v in self.vv]
    olabels     = [v.name.replace('_','') for v in self.observables]

    args_sts     = []
    args_obs        = []

    if wannasee is not None and wannasee is not 'all':
        for v_raw in wannasee:
            v   = v_raw.replace('_','') 
            if v in slabels:
                args_sts.append(slabels.index(v))
            elif v in olabels:
                args_obs.append(olabels.index(v))
            else:
                raise Exception("Variable %s neither in states nor observables. You might don't want to call self.get_sys() with the 'reduce_sys = False' argument. Note that underscores '_' are discarged." %v)

    st_vec          = np.zeros(len(self.vv))

    Y   = []
    K   = []
    L   = []
    superflag   = False

    st  = time.time()

    for t in range(30):

        shk_vec     = np.zeros(len(self.shocks))
        for vec in shocklist: 
            if vec[2] == t:

                shock       = vec[0]
                shocksize   = vec[1]

                shock_arg           = [v.name for v in self.shocks].index(shock)
                shk_vec[shock_arg]  = shocksize

                shk_process     = (self.SIG @ shk_vec).nonzero()

                for shk in shk_process:
                    args_sts += list(shk)
                
        st_vec, (l,k), flag     = self.t_func(st_vec, shk_vec, return_k = True, use_bruite = use_bruite)

        if flag: 
            superflag   = True

        Y.append(st_vec)
        K.append(k)
        L.append(l)

    if superflag and show_warnings:
        warnings.warn('Numerical errors in boehlgorithm, did not converge')

    Y   = np.array(Y)
    K   = np.array(K)
    L   = np.array(L)

    care_for_sts    = np.unique(args_sts)
    care_for_obs    = np.unique(args_obs)

    if wannasee is None:
        Z       = (self.hx[0] @ Y.T).T + self.hx[1]
        tt      = ~fast0(Z-Z.mean(axis=0),0)
        llabels = list(np.array(self.observables)[tt])+list(self.vv[care_for_sts])
        X2      = Y[:,care_for_sts]
        X       = np.hstack((Z[:,tt],X2))
    elif wannasee is 'all':
        tt      = ~fast0(Y-Y.mean(axis=0),0)
        llabels = list(self.vv[tt])
        X       = Y[:,tt]
    else:
        llabels = list(self.vv[care_for_sts])
        X       = Y[:,care_for_sts]
        if care_for_obs.size:
            llabels = list(np.array(self.observables)[care_for_obs]) + llabels
            Z       = ((self.hx[0] @ Y.T).T + self.hx[1])[:,care_for_obs]
            X       = np.hstack((Z,X))

    if verbose:
        print('[irfs:]'.ljust(15, ' ')+'Simulation took ', time.time() - st, ' seconds.')

    labels  = []
    for l in llabels:
        if not isinstance(l, str):
            l   = l.name
        labels.append(l)

    return X, labels, (Y, K, L)


def simulate(self, eps = None, mask = None, initial_state = None, use_bruite = 1, verbose = False, show_warnings = True, return_flag = False):
    """
        eps: shock innovations of shape (T, n_eps)
    """

    if eps is None:
        eps     = self.res.copy()

    if mask is not None:
        eps     = np.where(np.isnan(mask), eps, mask*eps)

    if initial_state is None:
        if hasattr(self, 'filtered_X'):
            st_vec          = self.filtered_X[0]
        else:
            st_vec          = np.zeros(len(self.vv))
    else:
        st_vec          = initial_state

    X   = [st_vec]
    K   = [0]
    L   = [0]
    superflag   = False

    if verbose:
        st  = time.time()

    for eps_t in eps:

        st_vec_new, (l,k), flag     = self.t_func(st_vec, noise=eps_t, return_k=True, use_bruite = use_bruite)

        if flag: 
            superflag   = True

        st_vec  = st_vec_new 

        X.append(st_vec)
        K.append(k)
        L.append(l)

    X   = np.array(X)
    K   = np.array(K)
    L   = np.array(L)

    self.simulated_X    = X

    if verbose:
        print('[simulate:]'.ljust(15, ' ')+'Simulation took ', time.time() - st, ' seconds.')

    if superflag and show_warnings:
        print('[simulate:]'.ljust(15, ' ')+'Numerical errors in boehlgorithm during simulation, did not converge')

    if return_flag:
        return X, np.expand_dims(K, 2), superflag

    return X, np.expand_dims(K, 2)


def linear_representation(self, l=0, k=0):

    N, A, J, cx, b, x_bar  = self.sys

    if not k:
        l   = 1

    mat, term    = self.precalc_mat

    dim_x  = J.shape[0]

    return mat[1, 0, 1][dim_x:]


def t_func(self, state, noise = None, return_flag = True, return_k = False, linear = False, use_bruite = False):

    newstate   = state.copy()

    if noise is not None:
        newstate   += self.SIG @ noise
    newstate, (l,k), flag   = boehlgorithm(self, newstate, linear = linear, use_bruite = use_bruite)

    if return_k: 	    return newstate, (l,k), flag
    elif return_flag:   return newstate, flag
    else: 			    return newstate

def o_func(self, state):
    """
    observation function
    (probably due for removal)
    """
    return self.hx[0] @ state + self.hx[1]
