#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import time
from .parser import DSGE as dsge
from numba import njit

aca = np.ascontiguousarray


@njit(nogil=True, cache=True)
def preprocess_jit(vals, l_max, k_max):

    # these must be the real max values, not only the size of the matrices
    l_max += 1
    k_max += 1

    N, A, J, cx, b, x_bar = vals

    dim_x, dim_y = J.shape
    dim_v = dim_y - dim_x

    s_max = l_max + k_max

    mat = np.empty((l_max, k_max, s_max, dim_y, dim_y-dim_x))
    term = np.empty((l_max, k_max, s_max, dim_y))
    core_mat = np.empty((l_max, s_max, dim_y, dim_y))
    core_term = np.empty((s_max, dim_y))

    core_mat[0, 0, :] = np.eye(dim_y)
    res = np.zeros((dim_y, dim_y))

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

            core = -nl.inv(JN[:, :dim_x])

            SS_mat = core @ aca(JN[:, dim_x:])
            SS_term = core @ sterm

            for s in range(s_max):

                doit = True
                if s > l:
                    l0 = l
                    if s > l+k+1:
                        doit = False
                    elif s == l+k+1:
                        k0 = k
                        s0 = 1
                    else:
                        k0 = s-l
                        s0 = 0
                else:
                    l0 = s
                    k0 = 0
                    s0 = 0

                if doit:

                    matrices = core_mat[l0, k0]
                    oterm = core_term[k0]

                    fin_mat = aca(matrices[:, :dim_x]) @ SS_mat + aca(matrices[:, dim_x:])
                    fin_term = aca(matrices[:, :dim_x]) @ SS_term + oterm

                    mat[l, k, s], term[l, k, s] = core_mat[s0,
                                                           0] @ fin_mat, core_mat[s0, 0] @ fin_term

    return mat, term


def preprocess(self, l_max=4, k_max=17, verbose=False):
    st = time.time()
    self.precalc_mat = preprocess_jit(self.sys, l_max, k_max)
    if verbose:
        print('[preprocess:]'.ljust(
            15, ' ')+'Preproceccing finished within %s s.' % np.round((time.time() - st), 3))


@njit(nogil=True, cache=True)
def LL_jit(l, k, s, v, mat, term):

    return mat[l, k, s] @ v + term[l, k, s]


@njit(nogil=True, cache=True)
def boehlgorithm_jit(N, A, J, cx, b, x_bar, v, mat, term, max_cnt):

    l_max = mat.shape[0] - 1
    k_max = mat.shape[1] - 1

    flag = 0
    l, k = 0, 0

    # check if (0,0) is a solution
    while b @ LL_jit(l, 0, l, v, mat, term) - x_bar > 0:
        if l == l_max:
            break
        l += 1

    # check if (0,0) is a solution
    if l < l_max:
        # needs to be wrapped so that both loops can be exited at once
        l, k = bruite_wrapper(b, x_bar, v, mat, term)

        # if still no solution, use approximation
        if l == 999:
            # set error flag 'no solution'
            flag = 1
            l, k = 0, 0
            while b @ LL_jit(l, k, l+k, v, mat, term) - x_bar < 0:
                if k == k_max:
                    # set error flag 'no solution + k_max reached'
                    flag = 3
                    break
                k += 1

    # either l or k must be > 0
    if not k:
        l = 1
    v_new = LL_jit(l, k, 1, v, mat, term)[J.shape[0]:]

    return v_new, (l, k), flag


@njit(nogil=True, cache=True)
def bruite_wrapper(b, x_bar, v, mat, term):

    l_max = mat.shape[0] - 1
    k_max = mat.shape[1] - 1

    for l in range(l_max):
        for k in range(1, k_max):
            if l:
                if b @ LL_jit(l, k, 0, v, mat, term) - x_bar < 0:
                    continue
                if b @ LL_jit(l, k, l-1, v, mat, term) - x_bar < 0:
                    continue
            if b @ LL_jit(l, k, k+l, v, mat, term) - x_bar < 0:
                continue
            if b @ LL_jit(l, k, l, v, mat, term) - x_bar > 0:
                continue
            if b @ LL_jit(l, k, k+l-1, v, mat, term) - x_bar > 0:
                continue
            return l, k

    return 999, 999


def boehlgorithm(self, v, max_cnt=4e1, linear=False):

    if not linear:

        if not hasattr(self, 'precalc_mat'):
            self.preprocess(verbose=False)

        # numba does not like tuples of numpy arrays
        mat, term = self.precalc_mat
        N, A, J, cx, b, x_bar = self.sys

        return boehlgorithm_jit(N, A, J, cx, b, x_bar, v, mat, term, max_cnt)

    else:

        if not hasattr(self, 'precalc_mat'):
            self.preprocess(l_max=1, k_max=1, verbose=False)

        mat, term = self.precalc_mat
        dim_x = self.sys[2].shape[0]

        return LL_jit(1, 0, 1, v, *self.precalc_mat)[dim_x:], (0, 0), 0
