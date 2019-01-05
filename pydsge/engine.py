#!/bin/python2
# -*- coding: utf-8 -*-

directory = '/home/gboehl/repos/'
import os, sys, importlib, time
for i in os.listdir(directory):
    sys.path.append(directory+i)

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
from .parser import DSGE as dsge
from numba import njit

@njit(cache=True)
def geom_series(M, n):
    res  = np.zeros(M.shape)
    for i in range(n):
        gs_add(res,nl.matrix_power(M,i))
    return res

@njit(cache=True)
def gs_add(A, B):
	for i in range(len(A)):
		for j in range(len(A)):
			A[i][j] += B[i][j]

@njit(cache=True)
def preprocess_jit(vals, l_max, k_max):

    N, A, J, cx, b, x_bar   = vals

    dim_x, dim_y    = J.shape
    dim_v           = dim_y - dim_x

    s_max 	= l_max + k_max

    mat         = np.empty((l_max, k_max, s_max, dim_y, dim_y-dim_x))
    term        = np.empty((l_max, k_max, s_max, dim_y))
    core_mat    = np.empty((l_max, s_max, dim_y, dim_y))
    core_term   = np.empty((l_max, s_max, dim_y))

    for aa in range(l_max):
        for bb in range(s_max):
            core_mat[aa,bb,:], core_term[aa,bb,:]   = create_core(vals[:4], aa, bb)

    for ll in range(l_max):
        for ss in range(s_max):
            for kk in range(k_max):
                mat[ll,kk,ss], term[ll,kk,ss]   = create_finish(vals[:4], ll, kk, ss, core_mat, core_term)

    return mat, term


def preprocess(self, l_max = 4, k_max = 20, info = False):
    st  = time.time()
    self.precalc_mat    = preprocess_jit(self.sys, l_max, k_max)
    if info == 1: 
        print('Preproceccing finished within %s s.' % np.round((time.time() - st), 3))


@njit(cache=True)
def create_core(vals, a, b):

    N, A, J, cx     = vals
    dim_x, dim_y    = J.shape

    term			= geom_series(N, b) @ cx
    if a:
        N_k 		    = nl.matrix_power(N,b)
        A_k             = nl.matrix_power(A,a)
        matrices 		= N_k @ A_k
    elif b:
        N_k 		    = nl.matrix_power(N,b)
        matrices 		= N_k
    else:
        matrices    = np.eye(dim_y)

    return matrices, term


@njit(cache=True)
def create_finish(vals, l, k, s, core_mat, core_term):

    N, A, J, cx     = vals
    dim_x, dim_y    = J.shape

    JN      = J @ core_mat[l,k]
    term    = J @ core_term[l,k]

    core        = -nl.inv(JN[:,:dim_x]) 

    SS_mat, SS_term     = core @ JN[:,dim_x:], core @ term

    k0 		= max(s-l, 0)
    l0 		= min(l, s)

    matrices 	= core_mat[l0,k0]
    term 		= core_term[l0,k0]

    fin_mat     = matrices[:,:dim_x] @ SS_mat + matrices[:,dim_x:]
    fin_term    = matrices[:,:dim_x] @ SS_term + term 

    return fin_mat, fin_term


@njit(cache=True)
def LL_pp(l, k, s, v, mat, term):

    return mat[l, k, s] @ v + term[l, k, s]


@njit(cache=True)
def SS_jit(vals, l, k, v):

    N, A, J, cx  = vals
    dim_x, dim_y    = J.shape

    term 		= J @ geom_series(N, k) @ cx
    if l:
        N_k 		= nl.matrix_power(N,k)
        A_k         = nl.matrix_power(A,l)
        JN			= J @ N_k @ A_k
    elif k:
        N_k 		= nl.matrix_power(N,k-1)
        JN			= J @ N_k @ N
    else:
        JN    = J

    core        = -nl.inv(JN[:,:dim_x]) 

    return core @ JN[:,dim_x:] @ v + core @ term 


@njit(cache=True)
def LL_jit(l, k, s, v, vals):

    N, A, J, cx     = vals
    dim_x, dim_y    = J.shape

    k0 		= max(s-l, 0)
    l0 		= min(l, s)
    if l0:
        N_k 		    = nl.matrix_power(N,k0)
        A_k             = nl.matrix_power(A,l0)
        matrices 		= N_k @ A_k
    elif k0:
        N_k 		    = nl.matrix_power(N,k0)
        matrices 		= N_k
    else:
        matrices    = np.eye(dim_y)
    term			= geom_series(N, k0) @ cx

    return matrices[:,:dim_x] @ SS_jit(vals, l, k, v) + matrices[:,dim_x:] @ v + term


@njit(cache=True)
def boehlgorithm_pp(N, A, J, cx, b, x_bar, v, mat, term, max_cnt):

    dim_x, dim_y    = J.shape

    l, k 		= 0, 0
    l1, k1 		= 1, 1

    l_max   = mat.shape[0] - 1
    k_max   = mat.shape[1] - 1

    cnt     = 0
    flag    = False
    while (l, k) != (l1, k1):
        if cnt  > max_cnt:
            flag    = 1
            break
        l1, k1 		= l, k
        if l: l -= 1
        while b @ LL_pp(l, k, l, v, mat, term) - x_bar > 0:
            if l >= l_max:
                l = 0
                break
            l 	+= 1
        if (l) == (l1):
            if k: k -= 1
            while b @ LL_pp(l, k, l+k, v, mat, term) - x_bar < 0: 
                k +=1
                if k >= k_max:
                    flag    = 2
                    break
        cnt += 1
    if l:
        if b @ LL_pp(l, k, 0, v, mat, term) - x_bar < 0: 
            print('It is happening..!')
            l   = 0

    if not k: l = 1
    v_new 	= LL_pp(l, k, 1, v, mat, term)[dim_x:]

    return v_new, (l, k), flag


@njit(cache=True)
def boehlgorithm_jit(vals, v, max_cnt, k_max = 20, l_max = 20):

    N, A, J, cx, b, x_bar   = vals
    dim_x, dim_y    = J.shape
    
    l, k 		= 0, 0
    l1, k1 		= 1, 1

    cnt     = 0
    flag    = False
    while (l, k) != (l1, k1):
        cnt += 1
        if cnt  > max_cnt:
            flag    = 1
            break
        l1, k1 		= l, k
        if l: l -= 1
        while b @ LL_jit(l, k, l, v, vals[:4]) - x_bar > 0:
            if l > l_max:
                l       = 0
                break
            l 	+= 1
        if (l) == (l1):
            if k: k 		-= 1
            while b @ LL_jit(l, k, l+k, v, vals[:4]) - x_bar < 0: 
                k +=1
                if k > k_max:
                    flag    = 2
                    break
        cnt += 1

    if not k: l = 1
    v_new 	= LL_jit(l, k, 1, v, vals[:4])[dim_x:]

    return v_new, (l, k), flag


def boehlgorithm(model_obj, v, max_cnt = 5e1, linear = False):

    if linear:
        dim_x   = model_obj.sys[2].shape[0]

        return LL_jit(1, 0, 1, v, model_obj.sys[:4])[dim_x:], (0, 0), 0

    elif hasattr(model_obj, 'precalc_mat'):

        ## numba does not like tuples of numpy arrays
        mat, term                   = model_obj.precalc_mat
        N, A, J, cx, b, x_bar       = model_obj.sys

        return boehlgorithm_pp(N, A, J, cx, b, x_bar, v, mat, term, max_cnt)
    else:
        return boehlgorithm_jit(model_obj.sys, v, max_cnt)

dsge.preprocess   = preprocess
