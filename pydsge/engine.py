#!/bin/python2
# -*- coding: utf-8 -*-

# directory = '/home/gboehl/repos/'
# import os, sys, importlib, time
# for i in os.listdir(directory):
    # sys.path.append(directory+i)

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import time
from .parser import DSGE as dsge
from numba import njit

@njit(nogil=True, cache=True)
def preprocess_jit(vals, l_max, k_max):

    N, A, J, cx, b, x_bar   = vals

    dim_x, dim_y    = J.shape
    dim_v           = dim_y - dim_x

    s_max 	= l_max + k_max

    mat         = np.empty((l_max, k_max, s_max, dim_y, dim_y-dim_x))
    term        = np.empty((l_max, k_max, s_max, dim_y))
    core_mat    = np.empty((l_max, s_max, dim_y, dim_y))
    core_term   = np.empty((s_max, dim_y))

    core_mat[0,0,:] = np.eye(dim_y)
    res             = np.zeros((dim_y,dim_y))

    for s in range(s_max):

        if s:
            core_mat[0,s,:]   = core_mat[0,s-1,:] @ N
            res     += core_mat[0,s-1,:]

        core_term[s,:]   = res @ cx

        for l in range(1,l_max):

            core_mat[l,s,:]   = core_mat[l-1,s,:] @ A

    for l in range(l_max):
        for k in range(k_max):

            JN      = J @ core_mat[l,k]
            sterm   = J @ core_term[k]

            core    = -nl.inv(JN[:,:dim_x]) 

            SS_mat, SS_term     = core @ JN[:,dim_x:], core @ sterm

            for s in range(s_max):

                doit    = True
                if s > l:
                    l0  = l
                    if s > l+k+1:
                        doit    = False
                    elif s == l+k+1:
                        k0  = k 
                        s0  = 1
                    else:
                        k0  = s-l
                        s0  = 0
                else:
                    l0  = s
                    k0  = 0
                    s0  = 0

                if doit:

                    matrices 	= core_mat[l0,k0]
                    oterm 		= core_term[k0]

                    fin_mat     = matrices[:,:dim_x] @ SS_mat + matrices[:,dim_x:]
                    fin_term    = matrices[:,:dim_x] @ SS_term + oterm 

                    mat[l,k,s], term[l,k,s]   = core_mat[s0,0] @ fin_mat, core_mat[s0,0] @ fin_term

    return mat, term


def preprocess(self, l_max = 5, k_max = 25, verbose = False):
    st  = time.time()
    self.precalc_mat    = preprocess_jit(self.sys, l_max, k_max)
    if verbose:
        print('[preprocess:]'.ljust(15, ' ')+'Preproceccing finished within %s s.' % np.round((time.time() - st), 3))


@njit(nogil=True, cache=True)
def LL_jit(l, k, s, v, mat, term):

    return mat[l, k, s] @ v + term[l, k, s]


@njit(nogil=True, cache=True)
def boehlgorithm_jit(N, A, J, cx, b, x_bar, v, mat, term, max_cnt, use_bruite):

    dim_x, dim_y    = J.shape

    l_max   = mat.shape[0] - 1
    k_max   = mat.shape[1] - 1

    flag    = 0

    if not use_bruite:
        l, k 		= 0, 0
        l1, k1 		= 1, 1

        cnt     = 0

        while (l, k) != (l1, k1):
            if cnt  > max_cnt:
                flag    = 1
                break
            l1, k1 		= l, k

            ## try lower l. If l is stable, loop will increment again and exit
            if l: 
                if b @ LL_jit(l, k, l-1, v, mat, term) - x_bar < 0:
                    l -= 1
            while b @ LL_jit(l, k, l, v, mat, term) - x_bar > 0:
                if l >= l_max:
                    l = 0
                    break
                l 	+= 1

            ## if l is stable, loop over k
            if (l) == (l1):
                if k: 
                    if b @ LL_jit(l, k, l+k-1, v, mat, term) - x_bar > 0: 
                        k -= 1
                while b @ LL_jit(l, k, l+k, v, mat, term) - x_bar < 0: 
                    k +=1
                    if k >= k_max:
                        flag    = 2
                        break
            cnt += 1

        ## if l, the first period should always be unconstrained
        if l: 
            if b @ LL_jit(l, k, 0, v, mat, term) - x_bar < 0: 
                flag    = 1

        if flag == 1:
            ## if there was no equilibirum, try to find it by bruit forcing
            l, k    = bruite(b, x_bar, v, mat, term)
            if l == -1:
                l, k    = 1, 0
            else:
                flag    = 0

    else:
        l   = 0
        k   = 0
        while b @ LL_jit(l, 0, l, v, mat, term) - x_bar > 0:
            if l == l_max: break
            l 	+= 1

        if l < l_max:
            l, k    = bruite(b, x_bar, v, mat, term)
            if l == -1:
                flag    = 1
                l, k    = 1, 0

    if not k:
        l = 1
    v_new 	= LL_jit(l, k, 1, v, mat, term)[dim_x:]

    return v_new, (l, k), flag


@njit(nogil=True, cache=True)
def bruite(b, x_bar, v, mat, term):

    l_max   = mat.shape[0] - 1
    k_max   = mat.shape[1] - 1

    for l in range(l_max):
        for k in range(k_max):
            yes     = True
            if k: 
                if b @ LL_jit(l, k, k+l-1, v, mat, term) - x_bar > 0:
                    yes     = False
                if yes: 
                    if b @ LL_jit(l, k, l, v, mat, term) - x_bar > 0:
                        yes     = False
            if l: 
                if yes:
                    if b @ LL_jit(l, k, l-1, v, mat, term) - x_bar < 0:
                        yes     = False
                if yes:
                    if b @ LL_jit(l, k, 0, v, mat, term) - x_bar < 0:
                        yes     = False
            if yes:
                if b @ LL_jit(l, k, k+l, v, mat, term) - x_bar < 0:
                    yes     = False
            if yes and k: 
                return (l, k)

    return (-1, -1)


def boehlgorithm(self, v, max_cnt = 4e1, linear = False, use_bruite = False):

    if not linear:

        if not hasattr(self, 'precalc_mat'):
            self.preprocess(self, verbose = False)

        ## numba does not like tuples of numpy arrays
        mat, term                   = self.precalc_mat
        N, A, J, cx, b, x_bar       = self.sys

        return boehlgorithm_jit(N, A, J, cx, b, x_bar, v, mat, term, max_cnt, use_bruite)

    else:

        if not hasattr(self, 'precalc_mat'):
            self.preprocess(self, l_max = 1, k_max = 1, verbose = False)

        mat, term       = self.precalc_mat
        dim_x           = self.sys[2].shape[0]

        return LL_jit(1, 0, 1, v, self.sys[:4])[dim_x:], (0, 0), 0



