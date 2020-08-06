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

aca = np.ascontiguousarray

# def gen_sys(self, par=None, reduce_sys=None, l_max=None, k_max=None, tol=1e-08, ignore_tests=False, verbose=False):
    # """Creates the transition function given a set of parameters. 

    # If no parameters are given this will default to the calibration in the `yaml` file.

    # Parameters
    # ----------
    # par : array or list, optional
        # The parameters to parse into the transition function. (defaults to calibration in `yaml`)
    # reduce_sys : bool, optional
        # If true, the state space is reduced. This speeds up computation.
    # l_max : int, optional
        # The expected number of periods *until* the constraint binds (defaults to 3).
    # k_max : int, optional
        # The expected number of periods for which the constraint binds (defaults to 17).
    # """

    # st = time.time()

    # reduce_sys = reduce_sys if reduce_sys is not None else self.fdict.get(
        # 'reduce_sys')
    # ignore_tests = ignore_tests if ignore_tests is not None else self.fdict.get(
        # 'ignore_tests')

    # if l_max is not None:
        # if l_max < 2:
            # print('[get_sys:]'.ljust(15, ' ') +
                  # ' `l_max` must be at least 2 (is %s). Correcting...' % l_max)
            # l_max = 2
        # # effective l_max is one lower because algorithm exists on l_max
        # l_max += 1

    # elif hasattr(self, 'lks'):
        # l_max = self.lks[0]
    # else:
        # l_max = 3

    # if k_max is not None:
        # pass
    # elif hasattr(self, 'lks'):
        # k_max = self.lks[1]
    # else:
        # k_max = 17

    # self.lks = [l_max, k_max]

    # self.fdict['reduce_sys'] = reduce_sys
    # self.fdict['ignore_tests'] = ignore_tests

    # par = self.p0() if par is None else list(par)
    # try:
        # ppar = self.pcompile(par)  # parsed par
    # except AttributeError:
        # ppar = self.compile(par)  # parsed par

    # self.par = par
    # self.ppar = ppar

    # if not self.const_var:
        # raise NotImplementedError('Package is only meant to work with OBCs')

    # vv_v = np.array([v.name for v in self.variables])
    # vv_x = np.array(self.variables)

    # dim_v = len(vv_v)

    # # obtain matrices
    # AA = self.AA(ppar)              # forward
    # BB = self.BB(ppar)              # contemp
    # CC = self.CC(ppar)              # backward
    # bb = self.bb(ppar).flatten().astype(float)  # constraint

    # # define transition shocks -> state
    # D = self.PSI(ppar)

    # # mask those vars that are either forward looking or part of the constraint
    # in_x = ~fast0(AA, 0) | ~fast0(bb[:dim_v])

    # # reduce x vector
    # vv_x2 = vv_x[in_x]
    # A1 = AA[:, in_x]
    # b1 = np.hstack((bb[:dim_v][in_x], bb[dim_v:]))

    # dim_x = len(vv_x2)

    # # define actual matrices
    # N = np.block([[np.zeros(A1.shape), CC], [
                 # np.eye(dim_x), np.zeros((dim_x, dim_v))]])

    # P = np.block([[-A1, -BB], [np.zeros((dim_x, dim_x)), np.eye(dim_v)[in_x]]])

    # c_arg = list(vv_x2).index(self.const_var)

    # # c contains information on how the constraint var affects the system
    # c1 = N[:, c_arg]
    # c_P = P[:, c_arg]

    # # get rid of constrained var
    # b2 = np.delete(b1, c_arg)
    # N1 = np.delete(N, c_arg, 1)
    # P1 = np.delete(P, c_arg, 1)
    # vv_x3 = np.delete(vv_x2, c_arg)
    # dim_x = len(vv_x3)

    # M1 = N1 + np.outer(c1, b2)

    # # solve using Klein's method
    # OME = re_bk(M1, P1, d_endo=dim_x)
    # J = np.hstack((np.eye(dim_x), -OME))

    # # desingularization of P
    # U, s, V = nl.svd(P1)

    # s0 = s < tol

    # P2 = U.T @ P1
    # N2 = U.T @ N1
    # c2 = U.T @ c1

    # # actual desingularization by iterating equations in M forward
    # P2[s0] = N2[s0]

    # # I could possible create auxiallary variables to make this work. Or I get the stuff directly from the boehlgo
    # if not fast0(c2[s0], 2) or not fast0(U.T[s0] @ c_P, 2):
        # raise NotImplementedError(
            # 'The system depends directly or indirectly on whether the constraint holds in the future or not.\n')

    # if verbose > 1:
        # print('[get_sys:]'.ljust(15, ' ') +
              # ' determinant of `P` is %1.2e.' % nl.det(P2))

    # if 'x_bar' in [p.name for p in self.parameters]:
        # x_bar = par[[p.name for p in self.parameters].index('x_bar')]
    # elif 'x_bar' in self.parafunc[0]:
        # pf = self.parafunc
        # x_bar = pf[1](par)[pf[0].index('x_bar')]
    # else:
        # print("Parameter `x_bar` (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        # x_bar = -1

    # try:
        # cx = nl.inv(P2) @ c2*x_bar
    # except ParafuncError:
        # raise SyntaxError(
            # "At least one parameter is a function of other parameters, and should be declared in `parafunc`.")

    # # create the stuff that the algorithm needs
    # N = nl.inv(P2) @ N2
    # A = nl.inv(P2) @ (N2 + np.outer(c2, b2))

    # out_msk = fast0(N, 0) & fast0(A, 0) & fast0(b2) & fast0(cx)
    # out_msk[-len(vv_v):] = out_msk[-len(vv_v):] & fast0(self.ZZ(ppar), 0)
    # # store those that are/could be reduced
    # self.out_msk = out_msk[-len(vv_v):].copy()

    # if not reduce_sys:
        # out_msk[-len(vv_v):] = False

    # s_out_msk = out_msk[-len(vv_v):]

    # if hasattr(self, 'P'):
        # if self.P.shape[0] < sum(~s_out_msk):
            # P_new = np.zeros((len(self.out_msk), len(self.out_msk)))
            # if P_new[~self.out_msk][:, ~self.out_msk].shape != self.P.shape:
                # print('[get_sys:]'.ljust(
                    # 15, ' ')+' Shape missmatch of P-matrix, number of states seems to differ!')
            # P_new[~self.out_msk][:, ~self.out_msk] = self.P
            # self.P = P_new
        # elif self.P.shape[0] > sum(~s_out_msk):
            # self.P = self.P[~s_out_msk][:, ~s_out_msk]

    # # add everything to the DSGE object
    # self.vv = vv_v[~s_out_msk]
    # self.vx = np.array([v.name for v in vv_x3])
    # self.dim_x = dim_x
    # self.dim_v = len(self.vv)

    # self.hx = self.ZZ(ppar)[:, ~s_out_msk], self.DD(ppar).squeeze()
    # self.obs_arg = np.where(self.hx[0])[1]

    # N2 = N[~out_msk][:, ~out_msk]
    # A2 = A[~out_msk][:, ~out_msk]
    # J2 = J[:, ~out_msk]

    # self.SIG = (BB.T @ D)[~s_out_msk]

    # self.sys = N2, A2, J2, cx[~out_msk], b2[~out_msk], x_bar

    # if verbose:
        # print('[get_sys:]'.ljust(15, ' ')+' Creation of system matrices finished in %ss.'
              # % np.round(time.time() - st, 3))

    # preprocess(self, self.lks[0], self.lks[1], verbose)

    # if not ignore_tests:
        # test_obj = self.precalc_mat[0][1, 0, 1]
        # test_con = eig(test_obj[-test_obj.shape[1]:]) > 1
        # if test_con.any():
            # raise ValueError(
                # 'Explosive dynamics detected: %s EV(s) > 1' % sum(test_con))

    # return


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

        A = A[dim:]
        B = B[dim:]
        C = C[dim:]

        if verbose:
            print('cutting off %s rows...' % dim)

        return (A, B, C, True)

    return (A, B, C, False)


def squeezer(RA0, RB0, RC0, S):

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
    
    A, N, J, D, cc, x_bar, ff, S, aux = self.sys

    dimp = J.shape[0]

    F = aux[:,:-self.ve.shape[0]]
    E = aux[:,-self.ve.shape[0]:]

    omg = -J[:,dimp:]

    Fp0 = F[:dimp]
    Ep0 = E[:dimp]

    Fq0 = F[dimp:]
    Eq0 = E[dimp:]

    Fp = omg @ Fq0
    Ep = omg @ Eq0

    Fx = A[:,:dimp] @ Fp + A[:,dimp:] @ Fq0 
    Ex = A[:,:dimp] @ Ep + A[:,dimp:] @ Eq0

    Fq = Fx[dimp:]
    Eq = Ex[dimp:]
    Fp1 = Fx[:dimp]
    Ep1 = Ex[:dimp]

    Fq1 = (A @ Fx)[dimp:]
    Eq1 = (A @ Ex)[dimp:]

    # translate back to y 
    Fy = S @ np.vstack((Fq1, Fp1, Fq, Fp, Fq0, Fp0))
    Ey = S @ np.vstack((Eq1, Ep1, Eq, Ep, Eq0, Ep0))

    self.lin_sys = Fy[:-self.ve.shape[0]], Ey[:-self.ve.shape[0]]

    return 


def gen_sys(self, par=None, l_max=None, k_max=None, tol = 1e-8, ignore_tests=False, verbose=True):

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
    self.fdict['ignore_tests'] = ignore_tests

    par = self.p0() if par is None else list(par)
    try:
        ppar = self.pcompile(par)  # parsed par
    except AttributeError:
        ppar = self.compile(par)  # parsed par

    self.par = par
    self.ppar = ppar

    if not self.const_var:
        # print('[get_sys:]'.ljust(15, ' ') + ' no constraint defined. Enabling only linear simulation.')
        # linear = True
        raise NotImplementedError('Package is only meant to work with OBCs')

    linear = False

    # start
    vv0 = np.array([v.name for v in self.variables])

    dimz = len(vv0)
    dimeps = len(self.shocks)
    dimy = dimz+dimeps

    AA0 = self.AA(ppar)              # forward
    BB0 = self.BB(ppar)              # contemp
    CC0 = self.CC(ppar)              # backward
    DD0 = -self.PSI(ppar).astype(float)
    fbx = self.bb(ppar).flatten().astype(float)  # constraint
    fd0 = -self.bb_PSI(ppar).flatten().astype(float)  # constraint
    ff0 = fbx[:dimz], fbx[dimz:], fd0

    ZZ = self.ZZ(ppar)

    if not linear:
        c_arg = list(vv0).index(str(self.const_var))

        AA0[:,[-1,c_arg]] = AA0[:,[c_arg,-1]]
        BB0[:,[-1,c_arg]] = BB0[:,[c_arg,-1]]
        CC0[:,[-1,c_arg]] = CC0[:,[c_arg,-1]]
        ff0[0][[-1,c_arg]] = ff0[0][[c_arg,-1]]
        ff0[1][[-1,c_arg]] = ff0[1][[c_arg,-1]]

        ZZ[:,[-1,c_arg]] = ZZ[:,[c_arg, -1]]
        vv0[[-1,c_arg]] = vv0[[c_arg,-1]]

    self.hx = ZZ, self.DD(ppar).squeeze()
    self.obs_arg = np.where(self.hx[0])[1]

    AA0 = np.pad(AA0,((0,dimeps),(0,dimeps)))
    BB0 = sl.block_diag(BB0,np.eye(dimeps))
    CC0 = np.block([[CC0,DD0],[np.zeros((dimeps,dimy))]])
    ff0 = np.pad(ff0[0], (0,dimeps)), np.hstack((ff0[1], ff0[2]))
    vv0 = np.hstack((vv0,self.shocks))

    ## find AUX
    AA1 = np.pad(AA0, ((0,1),(0,0)))
    BB1 = np.vstack((BB0, ff0[0]))
    CC1 = np.vstack((CC0, ff0[1]))

    A = AA1.copy()
    B = BB1.copy()
    C = CC1.copy()
    vv = vv0.copy()

    while True:
        A,B,C,flag = cutoff(A,B,C,verbose=verbose>1)
        if not flag:
            break

    A,B,C,vva,auxa = desingularize(A,B,C,vv,verbose=verbose>1)
    C,B,A,vvc,auxc = desingularize(C,B,A,vva,verbose=verbose>1)
    A1,B1,C1,_ = cutoff(A,B,C,verbose=verbose>1)

    inp = ~fast0(A,0)
    iny = fast0(B1,0)
    inq = ~fast0(C1,0)
    dimp = sum(inp)
    dimq = sum(inq)
    dimx = dimp+dimq

    ## find S
    RA0 = np.pad(AA0.copy(), ((0,0),(0,dimx)))
    RB0 = np.pad(BB0.copy(), ((0,0),(0,dimx)))
    RC0 = np.pad(CC0.copy(), ((0,0),(0,dimx)))

    S0 = np.block([[auxa,np.zeros((auxa.shape[0],auxc.shape[0]))], [auxc]])

    RA1, RB1, RC1, S1 = squeezer(RA0,RB0,RC0,S0)

    RA, RB, RC = RA1.copy(), RB1.copy(), RC1.copy()

    u,s,_ = nl.svd(np.hstack((RA[:,iny], RC[:,iny])), full_matrices=True)

    RA = u.T[sum(s > tol):] @ RA
    RB = u.T[sum(s > tol):] @ RB
    RC = u.T[sum(s > tol):] @ RC

    q,r = sl.qr(RB) 

    RA = q.T[:dimy] @ RA
    RB = q.T[:dimy] @ RB
    RC = q.T[:dimy] @ RC

    SI = nl.inv(RB[:,:dimy])
    SAP = -SI @ RA[:,inp]
    SAQ = -SI @ RA[:,inq]
    SBP = -SI @ RB[:,inp]
    SBQ = -SI @ RB[:,inq]
    SCP = -SI @ RC[:,inp]
    SCQ = -SI @ RC[:,inq]

    # S = SAP, SAQ, SBP, SBQ, SCP, SCQ
    S = np.hstack((SAQ, SAP, SBQ, SBP, SCQ, SCP))

    if verbose:
        print('max error in S', abs(SI @ RB[:,:dimy] - np.eye(SI.shape[0])).max())
        print('det & cond S', nl.det(RB[:,:dimy]), nl.cond(RB[:,:dimy]))

    ## find constraint sys
    A2, B2, C2 = RA1, RB1, RC1

    ynr = iny.copy() # "in y not r"
    ynr[dimz-1] = False

    u,s,_ = nl.svd(np.hstack((A2[:,~inp], C2[:,~inq], B2[:,ynr])), full_matrices=True)

    A2 = u.T[sum(s > tol):] @ A2
    B2 = u.T[sum(s > tol):] @ B2
    C2 = u.T[sum(s > tol):] @ C2

    # q,r = sl.qr(B2)
    q,r = shredder(B2)

    A2 = q.T[:dimx] @ A2
    B2 = q.T[:dimx] @ B2
    C2 = q.T[:dimx] @ C2

    fb0, fc0 = ff0

    ## translate constraint eq
    fb = np.vstack((np.pad(fb0, (0,dimx)), np.zeros_like(S1), S1))
    fc = np.vstack((np.pad(fc0, (0,dimx)), S1, np.zeros_like(S1)))

    u,s,_ = nl.svd(np.hstack((fc[:,~inq], fb[:,ynr])), full_matrices=True)

    fb = u.T[sum(s > tol):] @ fb
    fc = u.T[sum(s > tol):] @ fc

    q,r = sl.qr(fb)

    fb = q.T[0] @ fb
    fc = q.T[0] @ fc

    fb1 = -fb/fb[dimz-1]
    fc1 = -fc/fb[dimz-1]

    if ~fast0(fc1[inp],2):
        raise NotImplementedError('fc1 of p is non-zero')

    ff1 = fb1[inq], np.hstack((fb1[inp], fc1[inq]))

    A1 = A2[:dimx]
    B1 = B2[:dimx] + np.outer(B2[:dimx,dimz-1],fb1)
    C1 = C2[:dimx] + np.outer(B2[:dimx,dimz-1],fc1)

    N1 =  np.hstack((B1[:,inp],C1[:,inq]))
    P1 = -np.hstack((A1[:,inp],B1[:,inq]))
    N2 =  np.hstack((B2[:,inp],C2[:,inq]))
    P2 = -np.hstack((A1[:,inp],B2[:,inq]))
    D = S0[:,[list(vvc).index(s) for s in self.shocks]]

    if verbose:
        print('dets1:', nl.det(N1),nl.det(P2),np.shape(N2))
        print('max error in N1:', abs(nl.inv(N1) @ N1 - np.eye(N1.shape[0])).max(), nl.cond(N1))
        print('max error in P1:', abs(nl.inv(P1) @ P1 - np.eye(P1.shape[0])).max(), nl.cond(P1))
        print('dets2:', nl.det(N2),nl.det(P2),np.shape(N2))
        print('max error in N2:', abs(nl.inv(N2) @ N2 - np.eye(N2.shape[0])).max(), nl.cond(N2))
        print('max error in P2:', abs(nl.inv(P2) @ P2 - np.eye(P2.shape[0])).max(), nl.cond(P2))

    # simulate
    omg = re_bk(N1,P1,d_endo=sum(inp))

    J = np.hstack((np.eye(omg.shape[0]), -omg))

    A = nl.inv(P1) @ N1
    N = nl.inv(P2) @ N2
    cc = nl.inv(P2) @ B2[:,dimz-1]

    if 'x_bar' in [p.name for p in self.parameters]:
        x_bar = par[[p.name for p in self.parameters].index('x_bar')]
    elif 'x_bar' in self.parafunc[0]:
        pf = self.parafunc
        x_bar = pf[1](par)[pf[0].index('x_bar')]
    else:
        print("Parameter `x_bar` (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        x_bar = -1

    self.vv = vv[:-len(self.shocks)]
    self.nvar = len(self.vv)

    # precalculate eigenvectors here
    
    wa, vla = sl.eig(A)
    wn, vln = sl.eig(N)
    vra = nl.inv(vla)
    vrn = nl.inv(vln)

    self.evs = vra, wa, vla, vrn, wn, vln
    self.sys = A, N, J, D, cc, x_bar, ff1, S, aca(S0[:,:-dimx])

    gen_lin_sys(self)
    preprocess(self, self.lks, verbose)

    return 
