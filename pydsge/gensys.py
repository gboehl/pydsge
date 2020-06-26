#!/bin/python
# -*- coding: utf-8 -*-

"""contains functions related to (re)compiling the model with different parameters
"""

import time
import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
from grgrlib import fast0, eig, re_bk
from .engine import preprocess
from .stats import post_mean

try:
    from numpy.core._exceptions import UFuncTypeError as ParafuncError
except ModuleNotFoundError:
    ParafuncError = Exception

def gen_sys(self, par=None, reduce_sys=None, l_max=None, k_max=None, tol=1e-08, ignore_tests=False, verbose=False):
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

    reduce_sys = reduce_sys if reduce_sys is not None else self.fdict.get(
        'reduce_sys')
    ignore_tests = ignore_tests if ignore_tests is not None else self.fdict.get(
        'ignore_tests')

    if l_max is not None:
        if l_max < 2:
            print('[get_sys:]'.ljust(15, ' ') +
                  ' `l_max` must be at least 2 (is %s). Correcting...' % l_max)
            l_max = 2
        # effective l_max is one lower because algorithm exists on l_max 
        l_max += 1

    elif hasattr(self, 'lks'):
        l_max = self.lks[0]
    else:
        l_max = 3

    if k_max is not None:
        pass
    elif hasattr(self, 'lks'):
        k_max = self.lks[1]
    else:
        k_max = 17

    self.lks = [l_max, k_max]

    self.fdict['reduce_sys'] = reduce_sys
    self.fdict['ignore_tests'] = ignore_tests

    par = self.p0() if par is None else list(par)
    try:
        ppar = self.pcompile(par)  # parsed par
    except AttributeError:
        ppar = self.compile(par)  # parsed par

    self.par = par
    self.ppar = ppar

    if not self.const_var:
        raise NotImplementedError('Package is only meant to work with OBCs')

    vv_v = np.array([v.name for v in self.variables])
    vv_x = np.array(self.variables)

    dim_v = len(vv_v)

    # obtain matrices
    AA = self.AA(ppar)              # forward
    BB = self.BB(ppar)              # contemp
    CC = self.CC(ppar)              # backward
    bb = self.bb(ppar).flatten().astype(float)  # constraint

    # define transition shocks -> state
    D = self.PSI(ppar)

    # mask those vars that are either forward looking or part of the constraint
    in_x = ~fast0(AA, 0) | ~fast0(bb[:dim_v])

    # reduce x vector
    vv_x2 = vv_x[in_x]
    A1 = AA[:, in_x]
    b1 = np.hstack((bb[:dim_v][in_x], bb[dim_v:]))

    dim_x = len(vv_x2)

    # define actual matrices
    N = np.block([[np.zeros(A1.shape), CC], [
                 np.eye(dim_x), np.zeros((dim_x, dim_v))]])

    P = np.block([[-A1, -BB], [np.zeros((dim_x, dim_x)), np.eye(dim_v)[in_x]]])

    c_arg = list(vv_x2).index(self.const_var)

    # c contains information on how the constraint var affects the system
    c1 = N[:, c_arg]
    c_P = P[:, c_arg]

    # get rid of constrained var
    b2 = np.delete(b1, c_arg)
    N1 = np.delete(N, c_arg, 1)
    P1 = np.delete(P, c_arg, 1)
    vv_x3 = np.delete(vv_x2, c_arg)
    dim_x = len(vv_x3)

    M1 = N1 + np.outer(c1, b2)

    # solve using Klein's method
    OME = re_bk(M1, P1, d_endo=dim_x)
    J = np.hstack((np.eye(dim_x), -OME))

    # desingularization of P
    U, s, V = nl.svd(P1)

    s0 = s < tol

    P2 = U.T @ P1
    N2 = U.T @ N1
    c2 = U.T @ c1

    # actual desingularization by iterating equations in M forward
    P2[s0] = N2[s0]

    # I could possible create auxiallary variables to make this work. Or I get the stuff directly from the boehlgo
    if not fast0(c2[s0], 2) or not fast0(U.T[s0] @ c_P, 2):
        raise NotImplementedError(
            'The system depends directly or indirectly on whether the constraint holds in the future or not.\n')

    if verbose > 1:
        print('[get_sys:]'.ljust(15, ' ') +
              ' determinant of `P` is %1.2e.' % nl.det(P2))

    if 'x_bar' in [p.name for p in self.parameters]:
        x_bar = par[[p.name for p in self.parameters].index('x_bar')]
    elif 'x_bar' in self.parafunc[0]:
        pf = self.parafunc
        x_bar = pf[1](par)[pf[0].index('x_bar')]
    else:
        print("Parameter `x_bar` (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        x_bar = -1

    try:
        cx = nl.inv(P2) @ c2*x_bar
    except ParafuncError:
        raise SyntaxError(
            "At least one parameter is a function of other parameters, and should be declared in `parafunc`.")

    # create the stuff that the algorithm needs
    N = nl.inv(P2) @ N2
    A = nl.inv(P2) @ (N2 + np.outer(c2, b2))

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

    self.hx = self.ZZ(ppar)[:, ~s_out_msk], self.DD(ppar).squeeze()
    self.obs_arg = np.where(self.hx[0])[1]

    N2 = N[~out_msk][:, ~out_msk]
    A2 = A[~out_msk][:, ~out_msk]
    J2 = J[:, ~out_msk]

    self.SIG = (BB.T @ D)[~s_out_msk]

    self.sys = N2, A2, J2, cx[~out_msk], b2[~out_msk], x_bar

    if verbose:
        print('[get_sys:]'.ljust(15, ' ')+' Creation of system matrices finished in %ss.'
              % np.round(time.time() - st, 3))

    preprocess(self, self.lks[0], self.lks[1], verbose)

    if not ignore_tests:
        test_obj = self.precalc_mat[0][1, 0, 1]
        test_con = eig(test_obj[-test_obj.shape[1]:]) > 1
        if test_con.any():
            raise ValueError(
                'Explosive dynamics detected: %s EV(s) > 1' % sum(test_con))

    return

def desingularize(A, B, C, D, vv, verbose=False):
    """Reduces `A` to full row-rank, then creates auxilliary variables and adds them as static variables
"""
    eia = ~fast0(A, 1)
    q, s, _ = nl.svd((A[eia]), full_matrices=True)
    A[eia] = q.T @ A[eia]
    B[eia] = q.T @ B[eia]
    C[eia] = q.T @ C[eia]
    D[eia] = q.T @ D[eia]
    eia = np.where(~fast0(A, 1))[0]
    A12 = A[eia].copy()
    dim = A12.shape[0]
    A = np.zeros((A.shape[0] + dim, A.shape[1] + dim))
    A[eia, -dim:] = np.eye(dim)
    A12 = np.hstack((A12, -np.eye(dim)))
    B = np.pad(B, ((0, dim), (0, dim)))
    B[-dim:] = A12
    C = np.pad(C, ((0, dim), (0, dim)))
    D = np.pad(D, ((0, dim), (0, 0)))
    lags = [int(v[4:]) for v in vv if 'AUX_' in v]
    maxlag = max(lags) + 1 if lags else 0
    vv = np.hstack((vv, ['AUX_%s' % (maxlag + i) for i in range(dim)]))
    if verbose:
        print('Added %s auxilliary variables' % dim)
    return (A, B, C, D, vv, A12)

def cutoff(A, B, C, D, verbose=False):
    """Solves for and removes static variables
    """
    viu = fast0(A, 0) & ~fast0(B, 0) & fast0(C, 0)
    dim = sum(viu)
    if viu.any():
        enb = ~fast0(B, 1)
        Q, R = sl.qr(B[enb][:, viu])
        A[enb] = Q.T @ A[enb]
        B[enb] = Q.T @ B[enb]
        C[enb] = Q.T @ C[enb]
        D[enb] = Q.T @ D[enb]
        A = A[dim:]
        B = B[dim:]
        C = C[dim:]
        D = D[dim:]
        if verbose:
            print('cutting off %s rows...' % dim)
        return (A, B, C, D, True)
    return (A, B, C, D, False)

def squeezer(RA0, RB0, RC0, RD0, S):
    st = time.time()
    while True:
        oldshape = S.shape
        null = np.zeros_like(S)
        RA = np.vstack((RA0, S, null, null))
        RB = np.vstack((RB0, null, S, null))
        RC = np.vstack((RC0, null, null, S))
        RD = np.pad(RD0, ((0, 3 * S.shape[0]), (0, 0)))
        M = np.hstack((RA, RC, RD))
        vim = np.any(M, 0)
        eim = np.any(M, 1)
        u, s, v = nl.svd(M[eim][:, vim])
        SB = np.vstack((RB[(~eim)], u.T[(s < 1e-08)] @ RB[eim]))
        u, s, v = nl.svd(SB)
        S = u.T[(s > 1e-08)] @ SB
        if S.shape == oldshape:
            break

    return (RA, RB, RC, RD, S)
