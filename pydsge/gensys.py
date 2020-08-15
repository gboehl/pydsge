#!/bin/python
# -*- coding: utf-8 -*-

"""contains functions related to (re)compiling the model with different parameters
"""

import time
import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
from grgrlib import fast0, eig, re_bk, shredder
from .engine import preprocess
from .stats import post_mean

try:
    from numpy.core._exceptions import UFuncTypeError as ParafuncError
except ModuleNotFoundError:
    ParafuncError = Exception

# try:
    # ...
# except ParafuncError:
    # raise SyntaxError(
    # "At least one parameter is a function of other parameters, and should be declared in `parafunc`.")

aca = np.ascontiguousarray


def desingularize(A, B, C, vv, verbose=False):
    """Reduces `A` to full row-rank, then creates auxilliary variables and adds them as static variables
    """

    eia = ~fast0(A, 1)

    q, s, _ = nl.svd((A[eia]), full_matrices=True)
    A[eia] = q.T @ A[eia]
    B[eia] = q.T @ B[eia]
    C[eia] = q.T @ C[eia]

    eia = np.where(~fast0(A, 1))[0]
    A12 = A[eia].copy()
    dim = A12.shape[0]

    A = np.zeros((A.shape[0] + dim, A.shape[1] + dim))
    A[eia, -dim:] = np.eye(dim)
    A12 = np.hstack((A12, -np.eye(dim)))
    B = np.pad(B, ((0, dim), (0, dim)))
    B[-dim:] = A12
    C = np.pad(C, ((0, dim), (0, dim)))

    lags = [int(v[4:]) for v in vv if 'AUX_' in v]
    maxlag = max(lags) + 1 if lags else 0
    vv = np.hstack((vv, ['AUX_%s' % (maxlag + i) for i in range(dim)]))

    if verbose:
        print('Added %s auxilliary variables' % dim)

    return (A, B, C, vv, A12)


def cutoff(A, B, C, verbose=False):
    """cut-off reduction.

    Solves for and removes static variables.
    """

    viu = fast0(A, 0) & ~fast0(B, 0) & fast0(C, 0)
    dim = sum(viu)

    if viu.any():

        enb = ~fast0(B, 1)
        Q, R = sl.qr(B[enb][:, viu])

        A[enb] = Q.T @ A[enb]
        B[enb] = Q.T @ B[enb]
        C[enb] = Q.T @ C[enb]

        A = A[dim:]
        B = B[dim:]
        C = C[dim:]

        if verbose:
            print('cutting off %s rows...' % dim)

        return (A, B, C, True)

    return (A, B, C, False)


def pile(RA0, RB0, RC0, S):
    """Pile representation algorithm

    The resulting system matrices contain all atainable information on the full (y,x) system. S contains time-t-only information on this system (static relationships among variables).
    """

    st = time.time()

    while True:

        oldshape = S.shape
        null = np.zeros_like(S)

        RA = np.vstack((RA0, S, null, null))
        RB = np.vstack((RB0, null, S, null))
        RC = np.vstack((RC0, null, null, S))

        M = np.hstack((RA, RC))
        vim = np.any(M, 0)
        eim = np.any(M, 1)

        u, s, v = nl.svd(M[eim][:, vim])
        SB = np.vstack((RB[(~eim)], u.T[(s < 1e-08)] @ RB[eim]))

        u, s, v = nl.svd(SB, full_matrices=False)
        S = u.T[(s > 1e-08)] @ SB

        if S.shape == oldshape:
            break

    return (RA, RB, RC, S)


def gen_lin_sys(self):
    """Get a linear representation of the system. System transition and shock propagation can be obtained using only (cheap) standard LA operations
    """

    A, N, J, g, x_bar, _, _, T, aux = self.sys

    dimp = J.shape[0]

    F = aux[:, :-self.ve.shape[0]]
    E = aux[:, -self.ve.shape[0]:]

    omg = -J[:, dimp:]

    Fq0 = F[dimp:]
    Eq0 = E[dimp:]

    Fp = omg @ Fq0
    Ep = omg @ Eq0

    Fx = A[:, :dimp] @ Fp + A[:, dimp:] @ Fq0
    Ex = A[:, :dimp] @ Ep + A[:, dimp:] @ Eq0

    Fq = Fx[dimp:]
    Eq = Ex[dimp:]
    Fp1 = Fx[:dimp]
    Ep1 = Ex[:dimp]

    # translate back to y
    Fy = T @ np.vstack((Fp1, Fq, Fp, Fq0))
    Ey = T @ np.vstack((Ep1, Eq, Ep, Eq0))

    self.lin_sys = Fy[:-self.ve.shape[0]], Ey[:-self.ve.shape[0]]

    return


def gen_sys(self, par=None, l_max=None, k_max=None, tol=1e-8, verbose=True):
    """Generate system matrices expressed in the one-sided, first-order compressed dimensionality reduction given a set of parameters. 

    Details can be found in "Efficient Solution of Models with Occasionally Binding Constraints" (Gregor Boehl).
    If no parameters are given this will default to the calibration in the `yaml` file.

    Parameters
    ----------
    par : array or list, optional
        The parameters to parse into the transition function. (defaults to calibration in `yaml`)
    l_max : int, optional
        The expected number of periods *until* the constraint binds (defaults to 3).
    k_max : int, optional
        The expected number of periods for which the constraint binds (defaults to 17).
    tol : float, optional
    verbose : bool or int, optional
        Level of verbosity
    """

    st = time.time()

    # set default values of l_max & k_max
    if l_max is not None:
        if l_max < 2:
            print('[get_sys:]'.ljust(15, ' ') +
                  ' `l_max` must be at least 2 (is %s). Correcting...' % l_max)
            l_max = 2
        # effective l_max is one lower because algorithm exists on l_max
        l_max += 1
        # TODO: test if true

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

    self.lks = np.array([l_max, k_max])

    self.par = self.p0() if par is None else list(par)
    self.ppar = self.pcompile(self.par)  # parsed par

    if not self.const_var:
        raise NotImplementedError('Package is only meant to work with OBCs')

    # start
    vv0 = np.array([v.name for v in self.variables])

    # z-space is the space of the original variables
    dimz = len(vv0)
    dimeps = len(self.shocks)
    # y-space is the space of the original variables augmented by the shocks
    dimy = dimz+dimeps

    AA0 = self.AA(self.ppar)              # forward
    BB0 = self.BB(self.ppar)              # contemp
    CC0 = self.CC(self.ppar)              # backward
    DD0 = -self.PSI(self.ppar).astype(float)
    fbc = self.bb(self.ppar).flatten().astype(float)  # constraint
    fd0 = -self.bb_PSI(self.ppar).flatten().astype(float)  # constraint
    fb0 = fbc[:dimz]
    fc0 = fbc[dimz:]

    # observables from z
    ZZ0 = self.ZZ0(self.ppar)

    c_arg = list(vv0).index(str(self.const_var))

    # convention: constraint var is last var in z-representation
    AA0[:, [-1, c_arg]] = AA0[:, [c_arg, -1]]
    BB0[:, [-1, c_arg]] = BB0[:, [c_arg, -1]]
    CC0[:, [-1, c_arg]] = CC0[:, [c_arg, -1]]
    fb0[[-1, c_arg]] = fb0[[c_arg, -1]]
    fc0[[-1, c_arg]] = fc0[[c_arg, -1]]
    ZZ0[:, [-1, c_arg]] = ZZ0[:, [c_arg, -1]]
    vv0[[-1, c_arg]] = vv0[[c_arg, -1]]

    self.hx = ZZ0, self.ZZ1(self.ppar).squeeze()

    # create representation in y-space
    AA0 = np.pad(AA0, ((0, dimeps), (0, dimeps)))
    BB0 = sl.block_diag(BB0, np.eye(dimeps))
    CC0 = np.block([[CC0, DD0], [np.zeros((dimeps, dimy))]])
    fb0 = np.pad(fb0, (0, dimeps))
    fc0 = np.hstack((fc0, fd0))
    vv0 = np.hstack((vv0, self.shocks))

    # step 1: find auxiliary variables of CDR (= x-space)
    AA1 = np.pad(AA0, ((0, 1), (0, 0)))
    BB1 = np.vstack((BB0, fb0))
    CC1 = np.vstack((CC0, fc0))

    A = AA1.copy()
    B = BB1.copy()
    C = CC1.copy()
    vv = vv0.copy()

    # remove static endogenous variables (for debugging)
    # while True:
        # A, B, C, flag = cutoff(A, B, C, verbose=verbose > 1)
        # if not flag:
            # break

    # create auxiliary variables such that non-zero rows of A & C have full row rank
    A, B, C, vva, auxa = desingularize(A, B, C, vv, verbose=verbose > 1)
    C, _, A, vvc, auxc = desingularize(C, B, A, vva, verbose=verbose > 1)

    inp = ~fast0(A, 0)
    inq = ~fast0(C, 0)
    iny = ~inp & ~inq
    # forward looking variables in CDR
    dimp = sum(inp)
    # state variables in CDR
    dimq = sum(inq)
    # call both together "x-space"
    dimx = dimp+dimq

    # step 2: recover y-representation from x-representation
    RA0 = np.pad(AA0.copy(), ((0, 0), (0, dimx)))
    RB0 = np.pad(BB0.copy(), ((0, 0), (0, dimx)))
    RC0 = np.pad(CC0.copy(), ((0, 0), (0, dimx)))

    # stack y -> p and (y,p) -> q
    S0 = np.block([[auxa, np.zeros((auxa.shape[0], auxc.shape[0]))], [auxc]])

    # apply pile representation
    RA1, RB1, RC1, S1 = pile(RA0, RB0, RC0, S0)

    RA, RB, RC = RA1.copy(), RB1.copy(), RC1.copy()

    # use pile representation to remove all y(+1), q(+1) from A and all y(-1), p(-1) from C
    # u, s, _ = nl.svd(np.hstack((RA[:, iny], RC[:, iny])), full_matrices=True)
    u, s, _ = nl.svd(np.hstack((RA[:, ~inp], RC[:, ~inq])), full_matrices=True)

    RA = u.T[sum(s > tol):] @ RA
    RB = u.T[sum(s > tol):] @ RB
    RC = u.T[sum(s > tol):] @ RC

    q, r = sl.qr(RB)

    RA = q.T[:dimy] @ RA
    RB = q.T[:dimy] @ RB
    RC = q.T[:dimy] @ RC

    # invert system to get a representation from (x(+1), x, x(-1)) -> y
    TI = nl.inv(RB[:, :dimy])
    TAP = -TI @ RA[:, inp]
    # TAQ = -TI @ RA[:, inq]
    TBP = -TI @ RB[:, inp]
    TBQ = -TI @ RB[:, inq]
    TCQ = -TI @ RC[:, inq]
    # TCP = -TI @ RC[:, inp]

    # store in T
    # T = np.hstack((TAQ, TAP, TBQ, TBP, TCQ, TCP))
    T = np.hstack((TAP, TBQ, TBP, TCQ))

    if verbose:
        print('max error in T', abs(
            TI @ RB[:, :dimy] - np.eye(TI.shape[0])).max())
        print('det & cond T', nl.det(RB[:, :dimy]), nl.cond(RB[:, :dimy]))

    # step 3: find CDR of constraint system
    A2, B2, C2 = RA1, RB1, RC1

    ynr = iny.copy()  # "in y not r"
    ynr[dimz-1] = False

    u, s, _ = nl.svd(
        np.hstack((A2[:, ~inp], C2[:, ~inq], B2[:, ynr])), full_matrices=True)

    A2 = u.T[sum(s > tol):] @ A2
    B2 = u.T[sum(s > tol):] @ B2
    C2 = u.T[sum(s > tol):] @ C2

    q, r = shredder(B2)

    A2 = q.T[:dimx] @ A2
    B2 = q.T[:dimx] @ B2
    C2 = q.T[:dimx] @ C2

    # translate constraint eq
    fb = np.vstack((np.pad(fb0, (0, dimx)), S1))
    fc = np.vstack((np.pad(fc0, (0, dimx)), S1))

    q, r = shredder(fc[:, ~inq])
    fc = q.T[-1] @ fc

    q, r = shredder(fb[:, ynr])
    fb = q.T[-1] @ fb

    fb1 = -fb/fb[dimz-1]
    fc1 = -fc/fb[dimz-1]

    if ~fast0(fc1[inp], 2):
        raise NotImplementedError(
            'Constraint depends on future variables (fc1 of p is non-zero)')

    ff0 = fb1[inq]
    ff1 = np.hstack((fb1[inp], fc1[inq]))

    # done with the nasty part. Now find one-sided first-oder sys
    g1 = B2[:, dimz-1]
    B1 = B2 + np.outer(g1, fb1)
    C1 = C2 + np.outer(g1, fc1)

    P1 = -np.hstack((A2[:, inp], B1[:, inq]))
    N1 = np.hstack((B1[:, inp], C1[:, inq]))
    P2 = -np.hstack((A2[:, inp], B2[:, inq]))
    N2 = np.hstack((B2[:, inp], C2[:, inq]))

    # find linear RE solution for when constraint does not bind
    omg = re_bk(N1, P1, d_endo=sum(inp))
    J = np.hstack((np.eye(omg.shape[0]), -omg))

    # desingularization of P 
    U, s, V = nl.svd(P1)

    s0 = s < tol

    P1 = U.T @ P1
    N1 = U.T @ N1
    P2 = U.T @ P2
    N2 = U.T @ N2
    g1 = U.T @ g1

    # actual desingularization by iterating equations in M forward
    P1[s0] = N1[s0]
    P2[s0] = N2[s0]

    # future values must be read directly from the iterative output. Not yet implemented
    if not fast0(g1[s0], 2):
        raise NotImplementedError(
            'The system depends directly or indirectly on whether the constraint holds in the future or not.\n')

    if verbose:
        print('dets1:', nl.det(N1), nl.det(P1), np.shape(N2))
        print('max error in N1:', abs(nl.inv(N1) @ N1 -
                                      np.eye(N1.shape[0])).max(), nl.cond(N1))
        print('max error in P1:', abs(nl.inv(P1) @ P1 -
                                      np.eye(P1.shape[0])).max(), nl.cond(P1))
        print('dets2:', nl.det(N2), nl.det(P2), np.shape(N2))
        print('max error in N2:', abs(nl.inv(N2) @ N2 -
                                      np.eye(N2.shape[0])).max(), nl.cond(N2))
        print('max error in P2:', abs(nl.inv(P2) @ P2 -
                                      np.eye(P2.shape[0])).max(), nl.cond(P2))

    A = nl.inv(P1) @ N1
    N = nl.inv(P2) @ N2
    g = nl.inv(P2) @ g1

    # fix value of x_bar
    if 'x_bar' in [p.name for p in self.parameters]:
        x_bar = self.par[[p.name for p in self.parameters].index('x_bar')]
    elif 'x_bar' in self.parafunc[0]:
        pf = self.parafunc
        x_bar = pf[1](self.par)[pf[0].index('x_bar')]
    else:
        print("Parameter `x_bar` (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        x_bar = -1

    # finally add relevant stuff to the class
    self.vv = vv[:-len(self.shocks)]
    self.nvar = len(self.vv)

    # precalculate eigenvalues and eigenvectors
    # wa, vla = sl.eig(A)
    # wn, vln = sl.eig(N)
    # vra = nl.inv(vla)
    # vrn = nl.inv(vln)

    # self.evs = vra, wa, vla, vrn, wn, vln
    self.sys = A, N, J, g, x_bar, ff0, ff1, T, aca(S0[:, :-dimx])

    # also add simple linear representation
    gen_lin_sys(self)
    # preprocess all system matrices until (l_max, k_max)
    preprocess(self, verbose)

    if verbose:
        print('[get_sys:]'.ljust(15, ' ')+' Creation of system matrices finished in %ss.' %
              np.round(time.time() - st, 3))

    return
