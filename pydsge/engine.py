#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import time
from .parser import DSGE as dsge
from numba import njit

aca = np.ascontiguousarray


@njit(cache=True, nogil=True)
def preprocess_jit(sys, l_max, k_max):

    # these must be the real max values, not only the size of the matrices
    l_max += 1
    k_max += 1

    # N, A, J, cx, b, x_bar = vals
    A, N, J, D, cc, x_bar, ff, S, aux = sys
    cx = cc * x_bar

    dimp, dimq = J.shape
    s_max = l_max + k_max

    mat = np.empty((l_max, k_max, s_max, dimq, dimq))
    term = np.empty((l_max, k_max, s_max, dimq))
    core_mat = np.empty((l_max, s_max, dimq, dimq))
    core_term = np.empty((s_max, dimq))

    core_mat[0, 0, :] = np.identity(dimq)
    res = np.zeros((dimq, dimq))

    for s in range(s_max):

        if s:
            core_mat[0, s, :] = core_mat[0, s-1, :] @ N
            res += core_mat[0, s-1, :]

        core_term[s, :] = res @ cx

        for l in range(1, l_max):

            core_mat[l, s, :] = core_mat[l-1, s, :] @ A

    for l in range(l_max):

        for k in range(k_max):

            JN = J @ core_mat[l, k]
            sterm = J @ core_term[k]

            core = -nl.inv(JN[:, :dimp])

            # SS_mat = core @ aca(JN[:, dimp:])
            SS_mat = core @ aca(JN)
            SS_term = core @ sterm

            for s in range(s_max):

                doit = True
                l0 = s
                k0 = 0
                s0 = 0

                if s > l:
                    l0 = l
                    if s > l+k+1:
                        continue
                    elif s == l+k+1:
                        k0 = k
                        s0 = 1
                    else:
                        k0 = s-l
                        s0 = 0

                matrices = core_mat[l0, k0]
                oterm = core_term[k0]

                fin_mat = aca(matrices[:, :dimp]) @ SS_mat + aca(matrices)
                fin_term = aca(matrices[:, :dimp]) @ SS_term + oterm

                mat[l, k, s], term[l, k, s] = core_mat[s0, 0] @ fin_mat, core_mat[s0, 0] @ fin_term 

    return mat, term


def preprocess(self, lks, verbose):

    l_max, k_max = lks

    st = time.time()
    self.precalc_mat = preprocess_jit(self.sys, l_max, k_max)

    if verbose:
        print('[preprocess:]'.ljust(
            15, ' ')+' Preproceccing finished within %ss.' % np.round((time.time() - st), 3))

    return


# def boehlgorithm(self, v, max_cnt=4e1, linear=False):

    # if not linear:

        # if not hasattr(self, 'precalc_mat'):
            # self.preprocess(verbose=False)

        # # numba does not like tuples of numpy arrays
        # mat, term, bmat, bterm = self.precalc_mat
        # N, A, J, cx, b, x_bar = self.sys

        # return boehlgorithm_jit(N, A, J, cx, b, x_bar, v, mat, term, bmat, bterm, max_cnt)

    # else:

        # if not hasattr(self, 'precalc_mat'):
            # self.preprocess(l_max=1, k_max=1, verbose=False)

        # dim_x = self.sys[2].shape[0]

        # return (self.precalc_mat[0][1, 0, 1] @ v)[dim_x:], (0, 0), 0

def boehlgorithm(self, q, linear=False):

    if linear:
        raise NotImplementedError('linear not yet implemented')

    return boehlgorithm_jit(self.sys, self.lks, q)


@njit(nogil=True, cache=True)
def boehlgorithm_jit(sys, lks, q):

    A, N, J, D, cc, x_bar, ff, S, aux = sys

    l_max, k_max = lks
    dimp, dimx = J.shape

    flag = 0
    l, k = 0, 0
    done = False

    # check if (0,0) is a solution
    while check(sys, l, l, 0, q) - x_bar > 0:
        l += 1
        if l >= l_max:
            done = True
            break

    if not done:
        # check if (0,0) is a solution
        if l <= l_max:
            # needs to be wrapped so that both loops can be exited at once
            l, k = bruite_wrapper(sys, lks, q)

            # if still no solution, use approximation
            if l == 999:
                flag = 1
                l, k = 0, 0
                while check(sys, l+k, l, k, q) - x_bar > 0:
                    if k == k_max:
                        # set error flag 'no solution + k_max reached'
                        flag = 2

    return l,k, flag


@njit(cache=True, nogil=True)
def t_func_jit(sys, lks, state, set_k):

    A, N, J, D, cc, x_bar, ff, S, aux = sys

    mat, term = D
    l_max, k_max = lks

    dimx = J.shape[1]
    dimp = J.shape[0]
    dimq = dimx - dimp

    # translate y to x
    x = aca(aux) @ state

    p0 = x[:dimp]
    q0 = x[dimp:]

    if set_k == -1:
        l, k, flag = boehlgorithm_jit(sys, lks, aca(q0))
    else:
        l, k = int(not bool(set_k)), set_k
        flag = 0

    # p = solve_p(sys, l, k, q0)
    p = aca(mat[l,k,0][:dimp,dimp:]) @ q0 + term[l,k,0][:dimp]
    x = move_q(sys, 1, l, k, p, q0)

    q = x[dimp:]
    p1 = x[:dimp]

    q1 = move_q(sys, 2, l, k, p, q0)[dimp:]

    # translate back to y 
    y = S[0] @ p1 + S[1] @ q1 + S[2] @ p + S[3] @ q + S[4] @ p0 + S[5] @ q0 

    return y, l, k, flag



# def func_dispatch(self, full=False, max_cnt=4e1, njit_t_func=True):
    # ## dispatch is by default disabled and AFAIK not used in any of the codes

    # if not hasattr(self, 'precalc_mat'):
        # self.preprocess(verbose=False)

    # # numba does not like tuples of numpy arrays
    # mat, term, bmat, bterm = self.precalc_mat
    # N, A, J, cx, b, x_bar = self.sys
    # x2eps = self.SIG
    # hx0 = np.ascontiguousarray(self.hx[0].astype(float).T)
    # hx1 = self.hx[1]

    # def t_func_jit(state, noise=np.zeros(self.nobs)):

        # newstate = state.copy()

        # if full:
            # newstate += x2eps @ noise

        # res = boehlgorithm_jit(N, A, J, cx, b, x_bar,
                               # newstate, mat, term, bmat, bterm, max_cnt)
        # return res[0], res[2]

    # if njit_t_func:
        # t_func_jit = njit(t_func_jit)

    # self.t_func_jit = t_func_jit

    # if full:
        # noise0 = np.zeros(self.nobs)

        # @njit
        # def get_eps_jit(x, xp):
            # return (x - t_func_jit(xp, noise0)[0]) @ x2eps

        # @njit
        # def o_func_jit(state):
            # s = np.ascontiguousarray(state)
            # return s @ hx0 + hx1

        # self.o_func_jit = o_func_jit
        # self.get_eps_jit = get_eps_jit

        # return t_func_jit, o_func_jit, get_eps_jit

    # return t_func_jit


# @njit(cache=True, nogil=True)
# def solve_p(sys, l, k, q):

    # A, N, J, D, cc, x_bar, ff, S, aux = sys

    # dimx = J.shape[1]
    # dimp = J.shape[0]
    # dimq = dimx - dimp

    # mat = J @ nl.matrix_power(N,k) @ nl.matrix_power(A,l)
    # term = J @ nl.inv(np.eye(dimx) - N) @ (np.eye(dimx) - nl.matrix_power(N,k)) @ cc * x_bar
    # return -nl.inv(mat[:,:dimp]) @ (term + mat[:,dimp:] @ q)


@njit(cache=True, nogil=True)
def solve_p_cont(sys, evs, l,k,q):

    # A, N, J, D, cc, x_bar, ff, S, aux = self.sys
    # vra, wa, vla, vrn, wn, vln = self.evs

    A, N, JJ, D, cc, x_bar, ff, S, aux = sys
    vra, wa, vla, vrn, wn, vln = evs
    J = JJ.astype(np.complex128)

    dimx = J.shape[1]
    dimp = J.shape[0]

    mat = J @ vln*wn**k @ vrn @ vla*wa**l @ vra
    # term = J @ nl.inv(np.eye(dimx) - N) @ (np.eye(dimx) - vln*wn**k @ vrn) @ cc * x_bar
    # return -np.real(nl.inv(mat[:,:dimp]) @ (term + mat[:,dimp:] @ q))

    ding = J @ nl.inv(np.eye(dimx) - N).astype(np.complex128)

    term = ding @ (np.eye(dimx) - vln*wn**k @ vrn) @ (cc * x_bar).astype(np.complex128)
    # t2 = pre @ nl.inv(np.eye(dimx) - N) @ (np.eye(dimx) - vln*wn** min(max(s-l,0),k) @ vrn) @ cc * x_bar

    return -np.real(nl.inv(mat[:,:dimp]) @ (term + mat[:,dimp:] @ q.astype(np.complex128)))


@njit(cache=True, nogil=True)
def move_q(sys, s, l, k, p, q):

    A, N, J, D, cc, x_bar, ff, S, aux = sys
    mat, term = D

    dimp = J.shape[0]
    # dimx = J.shape[1]
    # dimq = dimx - dimp

    # pre = nl.matrix_power(A,max(s-k-l,0))
    # t1 = pre @ nl.matrix_power(N,min(max(s-l,0),k)) @ nl.matrix_power(A,min(s,l))
    # t2 = pre @ nl.inv(np.eye(dimx) - N) @ (np.eye(dimx) - nl.matrix_power(N,min(max(s-l,0),k))) @ cc * x_bar
    # print(t1.shape)
    # print(t2.shape)
    

    return aca(mat[l,k,s][:,:dimp]) @ p + aca(mat[l,k,s][:,dimp:]) @ q + term[l,k,s]


@njit(cache=True, nogil=True)
def move_q_cont(sys, evs, s, l, k, p, q):

    # A, N, J, D, cc, x_bar, ff, S, aux = self.sys
    # vra, wa, vla, vrn, wn, vln = self.evs
    A, N, J, D, cc, x_bar, ff, S, aux = sys
    vra, wa, vla, vrn, wn, vln = evs

    dimx = J.shape[1]
    dimp = J.shape[0]

    pre = vla*wa**max(s-k-l,0) @ vra
    t1 = pre @ vln*wn**min(max(s-l,0),k) @ vrn @ vla*wa**min(s,l) @ vra
    ding = pre @ nl.inv(np.eye(dimx) - N).astype(np.complex128)

    t2 = ding @ (np.eye(dimx) - vln*wn** min(max(s-l,0),k) @ vrn) @ (cc * x_bar).astype(np.complex128)
    # t2 = pre @ nl.inv(np.eye(dimx) - N) @ (np.eye(dimx) - vln*wn** min(max(s-l,0),k) @ vrn) @ cc * x_bar

    return np.real(t1[:,:dimp] @ p.astype(np.complex128) + t1[:,dimp:] @ q.astype(np.complex128) + t2)


@njit(nogil=True, cache=True)
def check(sys, s, l, k, q0):

    A, N, J, D, cc, x_bar, ff, S, aux = sys

    mat, term = D
    dimp = J.shape[0]

    p = aca(mat[l,k,0][:dimp,dimp:]) @ q0 + term[l,k,0][:dimp]

    x = move_q(sys, s, l, k, p, q0)
    q = move_q(sys, s+1, l, k, p, q0)[dimp:]

    q0 = x[dimp:]
    p = x[:dimp]


    return ff[0] @ p + ff[1] @ q + ff[2] @ q0


@njit(nogil=True, cache=True)
def bruite_wrapper(sys, lks, q):

    A, N, J, D, cc, x_bar, ff, S, aux = sys
    l_max, k_max = lks

    for l in range(l_max):
        for k in range(1, k_max):
            if l:
                if check(sys, 0, l, k, q) - x_bar < 0:
                    continue
                if l > 1:
                    if check(sys, l-1, l, k, q) - x_bar < 0:
                        continue
            if check(sys, k+l, l, k, q) - x_bar < 0:
                continue
            if check(sys, l, l, k, q) - x_bar > 0:
                continue
            if k > 1:
                if check(sys, k+l-1, l, k, q) - x_bar > 0:
                    continue
            return l, k

    return 999, 999


