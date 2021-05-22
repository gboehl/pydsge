#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import time
import sys
from numba import njit, prange

aca = np.ascontiguousarray
si_eps = sys.float_info.epsilon


@njit(cache=True, nogil=True)
def get_lam(omg, psi, S, T, V, W, h, l):

    dimp, dimq = omg.shape

    A = S if l else V
    B = T if l else W
    c = np.zeros(dimq) if l else h[:dimq]

    inv = nl.inv(A[:dimq, :dimq] + aca(A[:dimq, dimq:]) @ omg)
    lam = inv @ aca(B[:dimq, :dimq])
    xi = inv @ (c - aca(A[:dimq, dimq:]) @ psi)

    return lam, xi


@njit(cache=True, nogil=True)
def get_omg(omg, psi, lam, xi, S, T, V, W, h, l):

    dimp, dimq = omg.shape

    A = S if l else V
    B = T if l else W
    c = np.zeros(dimp) if l else h[dimq:]

    dum = (A[dimq:, :dimq] + aca(A[dimq:, dimq:]) @ omg)
    psi = dum @ xi + aca(A[dimq:, dimq:]) @ psi - c
    omg = dum @ lam - aca(B[dimq:, :dimq])

    return omg, psi


def preprocess_jittable(S, T, V, W, h, fq1, fp1, fq0, omg, lam, x_bar, l_max, k_max):
    """jitted preprocessing of system matrices until (l_max, k_max)
    """

    dimp, dimq = omg.shape
    l_max += 1
    k_max += 1

    T22 = T[dimq:, dimq:]
    if nl.cond(T22) > 1/si_eps:
        print('[preprocess:]'.ljust(15, ' ') +
              ' WARNING: at least one control indetermined')

    T22i = nl.inv(T22)
    T[dimq:] = T22i @ aca(T[dimq:])
    S[dimq:] = T22i @ aca(S[dimq:])

    W22 = W[dimq:, dimq:]
    W22i = nl.inv(W22)
    W[dimq:] = W22i @ aca(W[dimq:])
    V[dimq:] = W22i @ aca(V[dimq:])
    h[dimq:] = W22i @ aca(h[dimq:])

    pmat = np.empty((l_max, k_max, dimp, dimq))
    qmat = np.empty((l_max, k_max, dimq, dimq))
    pterm = np.empty((l_max, k_max, dimp))
    qterm = np.empty((l_max, k_max, dimq))

    pmat[0, 0] = omg
    pterm[0, 0] = np.zeros(dimp)
    qmat[0, 0] = lam
    qterm[0, 0] = np.zeros(dimq)

    for l in range(0, l_max):
        for k in range(0, k_max):

            if k or l:

                l_last = max(l-1, 0)
                k_last = k if l else max(k-1, 0)

                qmat[l, k], qterm[l, k] = get_lam(
                    pmat[l_last, k_last], pterm[l_last, k_last], S, T, V, W, h, l)
                pmat[l, k], pterm[l, k] = get_omg(
                    pmat[l_last, k_last], pterm[l_last, k_last], qmat[l, k], qterm[l, k], S, T, V, W, h, l)

    bmat = np.empty((5, l_max, k_max, dimq))
    bterm = np.empty((5, l_max, k_max))

    for l in range(0, l_max):
        for k in prange(0, k_max):

            # initialize local lam, xi to iterate upon
            lam = np.eye(dimq)
            xi = np.zeros(dimq)

            for s in range(l+k+1):

                l_loc = max(l-s, 0)
                k_loc = max(min(k, k+l-s), 0)

                y2r = fp1 @ pmat[l_loc, k_loc] + fq1 @ qmat[l_loc, k_loc] + fq0
                cr = fp1 @ pterm[l_loc, k_loc] + fq1 @ qterm[l_loc, k_loc]

                if s == 0:
                    bmat[0, l, k] = y2r @ lam
                    bterm[0, l, k] = cr + y2r @ xi
                if s == l-1:
                    bmat[1, l, k] = y2r @ lam
                    bterm[1, l, k] = cr + y2r @ xi
                if s == l:
                    bmat[2, l, k] = y2r @ lam
                    bterm[2, l, k] = cr + y2r @ xi
                if s == l+k-1:
                    bmat[3, l, k] = y2r @ lam
                    bterm[3, l, k] = cr + y2r @ xi
                if s == l+k:
                    bmat[4, l, k] = y2r @ lam
                    bterm[4, l, k] = cr + y2r @ xi

                lam = qmat[l_loc, k_loc] @ lam
                xi = qmat[l_loc, k_loc] @ xi + qterm[l_loc, k_loc]

    return pmat, qmat, pterm, qterm, bmat, bterm


preprocess_jit = njit(preprocess_jittable, cache=True, nogil=True)
preprocess_jit_parallel = njit(
    preprocess_jittable, cache=True, nogil=True, parallel=True)


@njit(cache=True, nogil=True, parallel=True)
def preprocess_tmats_jit(pmat, pterm, qmat, qterm, fq1, fp1, fq0, omg, l_max, k_max):
    """jitted preprocessing of system matrices until (l_max, k_max)
    """

    dimp, dimq = omg.shape
    l_max += 1
    k_max += 1

    tmat = np.empty((l_max + k_max, l_max, k_max, dimq))
    tterm = np.empty((l_max + k_max, l_max, k_max))

    for l in range(0, l_max):
        for k in prange(0, k_max):

            # initialize local lam, xi to iterate upon
            lam = np.eye(dimq)
            xi = np.zeros(dimq)

            for s in range(l_max + k_max):

                l_loc = max(l-s, 0)
                k_loc = max(min(k, k+l-s), 0)

                y2r = fp1 @ pmat[l_loc, k_loc] + fq1 @ qmat[l_loc, k_loc] + fq0
                cr = fp1 @ pterm[l_loc, k_loc] + fq1 @ qterm[l_loc, k_loc]
                tmat[s, l, k] = y2r @ lam
                tterm[s, l, k] = cr + y2r @ xi

                lam = qmat[l_loc, k_loc] @ lam
                xi = qmat[l_loc, k_loc] @ xi + qterm[l_loc, k_loc]

    return tmat, tterm


def preprocess(self, PU, MU, PR, MR, gg, fq1, fp1, fq0, parallel=False, verbose=False):
    """dispatcher to jitted preprocessing
    """

    l_max, k_max = self.lks
    omg, lam, x_bar = self.sys

    st = time.time()
    preprocess_jit_loc = preprocess_jit_parallel if parallel else preprocess_jit
    self.precalc_mat = preprocess_jit_loc(
        PU, MU, PR, MR, gg, fq1, fp1, fq0, omg, lam, x_bar, l_max, k_max)

    if verbose:
        print('[preprocess:]'.ljust(
            15, ' ')+' Preprocessing finished within %ss.' % np.round((time.time() - st), 5))

    return


def preprocess_tmats(self, fq1, fp1, fq0, verbose):
    """dispatcher to jitted preprocessing
    """

    l_max, k_max = self.lks
    omg, lam, x_bar = self.sys

    st = time.time()
    pmat, qmat, pterm, qterm, bmat, bterm = self.precalc_mat
    self.precalc_tmat = preprocess_tmats_jit(
        pmat, pterm, qmat, qterm, fq1, fp1, fq0, omg, l_max, k_max)

    if verbose:
        print('[preprocess_tmats:]'.ljust(
            15, ' ')+' Preprocessing finished within %ss.' % np.round((time.time() - st), 5))

    return


@njit(cache=True, nogil=True)
def t_func_jit(pmat, pterm, qmat, qterm, bmat, bterm, x_bar, hxp, hxq, hxc, state, shocks, set_l, set_k, x_space):
    """jitted transitiona function
    """

    s = np.hstack((state, shocks))

    if set_k == -1:
        # find (l,k) if requested
        l, k, flag = find_lk(bmat, bterm, x_bar, s)
    else:
        l, k = set_l, set_k
        flag = 0

    p = pmat[l, k] @ s + pterm[l, k]
    q = qmat[l, k] @ s + qterm[l, k]

    # either return p or obs
    if x_space:
        p_or_obs = hxp @ p + hxq @ q + hxc
    else:
        p_or_obs = p

    return p_or_obs, q, l, k, flag


@njit(nogil=True, cache=True)
def find_lk(bmat, bterm, x_bar, q):
    """iteration loop to find (l,k) given state q
    """

    _, l_max, k_max = bterm.shape

    flag = 0
    l, k = 0, 0

    # check if (0,0) is a solution
    while check_cnst(bmat, bterm, 2, l, 0, q) - x_bar > 0:
        l += 1
        if l == l_max:
            break

    # check if (0,0) is a solution
    if l < l_max:
        # needs to be wrapped so that both loops can be exited at once
        l, k = bruite_wrapper(bmat, bterm, x_bar, q)

        # if still no solution, use approximation
        if l == 999:
            flag = 1
            l, k = 0, 0
            loop_type = False

            while True:
                k += 1
                if not loop_type:
                    # first iterate until r < r_bar
                    loop_type = check_cnst(bmat, bterm, 4, 0, k, q) - x_bar < 0
                elif check_cnst(bmat, bterm, 4, 0, k, q) - x_bar > 0:
                    # then iterate until r > r_bar again
                    break
                if k == k_max-1:
                    # set error flag 'no solution + k_max reached'
                    flag = 2
                    break
    else:
        l = 0

    return l, k, flag


@njit(nogil=True, cache=True)
def bruite_wrapper(bmat, bterm, x_bar, q):
    """iterate over (l,k) until (l_max, k_max) and check if RE equilibrium
    """
    _, l_max, k_max = bterm.shape

    for l in range(l_max):
        for k in range(1, k_max):
            if l and check_cnst(bmat, bterm, 0, l, k, q) - x_bar < 0:
                continue
            if l > 1 and check_cnst(bmat, bterm, 1, l, k, q) - x_bar < 0:
                continue
            if check_cnst(bmat, bterm, 4, l, k, q) - x_bar < 0:
                continue
            if check_cnst(bmat, bterm, 2, l, k, q) - x_bar > 0:
                continue
            if k > 1 and check_cnst(bmat, bterm, 3, l, k, q) - x_bar > 0:
                continue
            return l, k

    return 999, 999


@njit(nogil=True, cache=True)
def check_cnst(bmat, bterm, s, l, k, q0):
    """constraint value in period s given CDR-state q0 under the assumptions (l,k)
    """
    return bmat[s, l, k] @ q0 + bterm[s, l, k]
