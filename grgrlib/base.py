#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import warnings
import pydsge
from numba import njit
import time
from .pyzlb import boehlgorithm


def eig(M):
    return np.sort(np.abs(nl.eig(M)[0]))[::-1]


def sorter(x, y):
    out     = np.empty_like(x, dtype=bool)
    out[:]  = False
    zero_y      = np.isclose(y,0)
    zero_x      = np.isclose(x,0)
    out[zero_y] = True
    out[zero_x] = True
    return out


def invertible_subm(A):
    """
    For an (m times n) matrix A with n > m this function finds the m columns that are necessary to construct a nonsingular submatrix of A.
    """

    q, r, p     = sl.qr(A, mode='economic', pivoting=True)

    res                     = np.zeros(A.shape[1], dtype=bool)
    res[p[:A.shape[0]]]     = True

    return res


@njit(cache=True)
def subt(A, B):
	res 	= A
	for i in range(len(A)):
		for j in range(len(A)):
			res[i][j] = A[i][j] - B[i][j]
	return res


def nul(n):
    return np.zeros((n,n))


def iuc(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    ## rounding is necessary to avoid false round-offs
    out[nonzero] = (abs(x[nonzero]/y[nonzero]).round(3) < 1.0)
    return out

def re_bc(N, d_endo):

    n   = N.shape[0]

    MM, PP, alp, bet, Q, Z    = sl.ordqz(N,np.eye(n),sort=iuc)

    if not fast0(Q @ MM @ Z.T - N, 2):
        # warnings.warn('Numerical errors in QZ')
        raise ValueError('Numerical errors in QZ')

    Z21     = Z.T[-d_endo:,:d_endo]
    Z22     = Z.T[-d_endo:,d_endo:]

    return -nl.inv(Z21) @ Z22


def fast0(A, mode=None):

    if mode == None:
        return np.isclose(A, 0)
    elif mode == 0:
        return np.isclose(A, 0).all(axis=0)
    elif mode == 1:
        return np.isclose(A, 0).all(axis=1)
    else:
        return np.allclose(A, 0)

def get_sys(self, par=None, care_for = None, info = False):

    self.python_other_matrices()

    if par is None:
        par     = self.p0()

    st  = time.time()

    if not self.const_var:
        warnings.warn('Code is only meant to work with OBCs')

    vv_v    = np.array(self.variables)
    vv_x    = np.array(self.variables)

    dim_v   = len(vv_v)

    ## obtain matrices from pydsge
    ## this can be further accelerated by getting them directly from the equations in pydsge
    AA  = self.AA(par)              # forward
    BB  = self.BB(par)              # contemp
    CC  = self.CC(par)              # backward
    b   = self.bb(par).flatten()    # constraint

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
    
    c_arg       = list(vv_x2).index(self.const_var)

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


    try:
        x_bar       = par[[p.name for p in self.parameters].index('x_bar')]
    except ValueError:
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

    if info == 1:
        print('Creation of system matrices finished in %ss.'
              % np.round(time.time() - st,3))

    var_str     = [ v.name for v in vv_v ]
    out_msk     = fast0(N, 0) & fast0(A, 0) & fast0(b2) & fast0(cx)
    out_msk[-len(vv_v):]    = out_msk[-len(vv_v):] & fast0(self.ZZ(par), 0)

    ## add everything to the DSGE object
    self.vv     = vv_v[~out_msk[-len(vv_v):]]

    self.observables    = self['observables']
    self.par    = par

    self.hx             = self.ZZ(par)[:,~out_msk[-len(vv_v):]], self.DD(par).squeeze()
    self.obs_arg        = np.where(self.hx[0])[1]

    N2  = N[~out_msk][:,~out_msk]
    A2  = A[~out_msk][:,~out_msk]
    J2  = J[:,~out_msk]

    self.SIG    = (BB.T @ D)[~out_msk[-len(vv_v):]]

    ## need to delete this zero, its just here cause I'm lazy
    self.sys 	= N2, A2, J2, 0, cx[~out_msk], b2[~out_msk], x_bar


def irfs(self, shocklist, wannasee = None):

    ## returns time series of impule responses 
    ## shocklist: takes list of tuples of (shock, size, timing) 
    ## wannasee: list of strings of the variables to be plotted and stored

    labels      = [v.name.replace('_','') for v in self.vv]
    if wannasee is not None:
        args_see    = [labels.index(v) for v in wannasee]
    else:
        args_see    = list(self.obs_arg)

    st_vec          = np.zeros(len(self.vv))

    Y   = []
    K   = []
    L   = []
    superflag   = False

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
                    args_see += list(shk)
                
        st_vec, (l,k), flag     = self.t_func(st_vec, shk_vec, True)

        if flag: 
            superflag   = True

        Y.append(st_vec)
        K.append(k)
        L.append(l)

    Y   = np.array(Y)
    K   = np.array(K)
    L   = np.array(L)

    care_for    = np.unique(args_see)

    X   = Y[:,care_for]

    if superflag:
        warnings.warn('Numerical errors in boehlgorithm, did not converge')

    return X, self.vv[care_for], (Y, K, L)


def simulate(self, EPS=None, initial_state=None):

    ## EPS: shock innovations of shape (T, n_eps)

    if EPS is None:
        EPS     = self.residuals

    # st_vec          = np.zeros_like(self.filtered_X[0])
    if initial_state is None:
        st_vec          = np.zeros(len(self.vv))
        X   = []
    else:
        st_vec          = initial_state
        X   = [st_vec]

    K   = []
    L   = []
    superflag   = False


    for eps in EPS:

        st_vec, (l,k), flag     = self.t_func(st_vec, noise=eps, return_k=True)

        if flag: 
            superflag   = True

        X.append(st_vec)
        K.append(k)
        L.append(l)

    X   = np.array(X)
    K   = np.array(K)
    L   = np.array(L)

    self.simulated_X    = X
    self.simulated_Z    = (self.hx[0] @ X.T).T + self.hx[1]

    if superflag:
        warnings.warn('Numerical errors in boehlgorithm, did not converge')

    return X, self.simulated_Z


from .plots import pplot 
from .estimation import bayesian_estimation
from .filtering import create_filter
from .filtering import run_filter

pydsge.DSGE.DSGE.get_sys            = get_sys
pydsge.DSGE.DSGE.irfs               = irfs
pydsge.DSGE.DSGE.simulate           = simulate
pydsge.DSGE.DSGE.create_filter      = create_filter
pydsge.DSGE.DSGE.run_filter         = run_filter
pydsge.DSGE.DSGE.bayesian_estimation    = bayesian_estimation
