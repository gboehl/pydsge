#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings
import os
import scipy.stats as ss
import scipy.optimize as so
from scipy.special import gammaln


def mc_error(x):
    means = np.mean(x, 0)
    return np.std(means) / np.sqrt(x.shape[0])


def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width

    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        # raise ValueError('Too few elements for interval calculation')
        warnings.warn('Too few elements for interval calculation.')

        return None, None

    else:

        min_idx = np.argmin(interval_width)
        hdi_min = x[min_idx]
        hdi_max = x[min_idx + interval_idx_inc]

        return hdi_min, hdi_max


def _hpd_df(x, alpha):

    cnames = ['hpd_{0:g}'.format(100 * alpha / 2),
              'hpd_{0:g}'.format(100 * (1 - alpha / 2))]

    sx = np.sort(x.flatten())
    hpd_vals = np.array(calc_min_interval(sx, alpha)).reshape(1, -1)

    return pd.DataFrame(hpd_vals, columns=cnames)


def summary(store, priors, tune=None, alpha=0.05, top=None, show_priors=True, min_col=80, swarm_mode=None):
    # in parts stolen from pymc3 because it looks really nice

    try:
        with os.popen('stty size', 'r') as rows_cols:
            cols = rows_cols.read().split()[1]
    except IndexError:
        cols = min_col + 1

    if swarm_mode is None:
        swarm_mode = False

    if swarm_mode or isinstance(store, tuple):
        min_col += 20
        swarm_mode = True
        xs, fs, ns = store
        ns = ns.squeeze()
        fas = (-fs[:, 0]).argsort()
        xs = xs[fas]
        fs = fs.squeeze()[fas]

    f_prs = [lambda x: pd.Series(x, name='distribution'),
             lambda x: pd.Series(x, name='pst_mean'),
             lambda x: pd.Series(x, name='sd/df')]

    funcs = [lambda x: pd.Series(np.mean(x), name='mean'),
             lambda x: pd.Series(np.std(x), name='sd'),
             lambda x: pd.Series(mc_error(x), name='mc_error'),
             lambda x: _hpd_df(x, alpha)]

    var_dfs = []
    for i, var in enumerate(priors):

        lst = []
        if show_priors and int(cols) > min_col:
            prior = priors[var]
            if len(prior) > 3:
                prior = prior[-3:]
            [lst.append(f(prior[j])) for j, f in enumerate(f_prs)]

        if swarm_mode:
            [lst.append(pd.Series(s[i], name=n))
             for s, n in zip(xs[:top], ns[:top])]
        else:
            vals = store[tune:, :, i]
            [lst.append(f(vals)) for f in funcs]
        var_df = pd.concat(lst, axis=1)
        var_df.index = [var]
        var_dfs.append(var_df)

    if swarm_mode:

        lst = []

        if show_priors and int(cols) > min_col:
            [lst.append(f('')) for j, f in enumerate(f_prs)]

        [lst.append(pd.Series(s, name=n)) for s, n in zip(fs[:top], ns[:top])]
        var_df = pd.concat(lst, axis=1)
        var_df.index = ['loglike']
        var_dfs.append(var_df)

    dforg = pd.concat(var_dfs, axis=0)

    return dforg


def mc_mean(trace, varnames):
    # in most parts just stolen from pymc3 because it looks really nice

    p_means = []

    for i, var in enumerate(varnames):
        vals = trace[:, :, i]
        p_means.append(np.mean(vals))

    return p_means


class InvGammaDynare(ss.rv_continuous):
    # probably not needed anymore

    name = 'inv_gamma_dynare'

    def _logpdf(self, x, s, nu):

        if x < 0:
            lpdf = -np.inf
        else:
            lpdf = np.log(2) - gammaln(nu/2) - nu/2*(np.log(2) -
                                                     np.log(s)) - (nu+1)*np.log(x) - .5*s/x**2

        return lpdf

    def _pdf(self, x, s, nu):
        return np.exp(self._logpdf(x, s, nu))


def inv_gamma_spec(mu, sigma):

    # directly stolen and translated from dynare/matlab. It is unclear to me what the sigma parameter stands for, as it does not appear to be the standard deviation. This is provided for compatibility reasons, I strongly suggest to use the inv_gamma distribution that simply takes mean / stdd as parameters.

    def ig1fun(nu): return np.log(2*mu**2) - np.log((sigma**2+mu**2)
                                                    * (nu-2)) + 2*(gammaln(nu/2)-gammaln((nu-1)/2))

    nu = np.sqrt(2*(2+mu**2/sigma**2))
    nu2 = 2*nu
    nu1 = 2
    err = ig1fun(nu)
    err2 = ig1fun(nu2)

    if err2 > 0:
        while nu2 < 1e12:  # Shift the interval containing the root.
            nu1 = nu2
            nu2 = nu2*2
            err2 = ig1fun(nu2)
            if err2 < 0:
                break
        if err2 > 0:
            raise ValueError(
                '[inv_gamma_spec:] Failed in finding an interval containing a sign change! You should check that the prior variance is not too small compared to the prior mean...')

    # Solve for nu using the secant method.
    while abs(nu2/nu1-1) > 1e-14:
        if err > 0:
            nu1 = nu
            if nu < nu2:
                nu = nu2
            else:
                nu = 2*nu
                nu2 = nu
        else:
            nu2 = nu
        nu = (nu1+nu2)/2
        err = ig1fun(nu)

    s = (sigma**2+mu**2)*(nu-2)

    if abs(np.log(mu)-np.log(np.sqrt(s/2))-gammaln((nu-1)/2)+gammaln(nu/2)) > 1e-7:
        raise ValueError(
            '[inv_gamma_spec:] Failed in solving for the hyperparameters!')
    if abs(sigma-np.sqrt(s/(nu-2)-mu*mu)) > 1e-7:
        raise ValueError(
            '[inv_gamma_spec:] Failed in solving for the hyperparameters!')

    return s, nu


def get_priors(priors):

    priors_lst = []
    initv = []
    lb = []
    ub = []

    print('Adding parameters to the prior distribution...')
    for pp in priors:

        dist = priors[str(pp)]

        if len(dist) == 3:
            initv.append(None)
            lb.append(None)
            ub.append(None)
            ptype = dist[0]
            pmean = dist[1]
            pstdd = dist[2]
        elif len(dist) == 6:
            if dist[0] == 'None':
                initv.append(None)
            else:
                initv.append(dist[0])
            lb.append(dist[1])
            ub.append(dist[2])
            ptype = dist[3]
            pmean = dist[4]
            pstdd = dist[5]
        else:
            raise NotImplementedError(
                'Shape of prior specification is unclear (!=3 & !=6).')

        # simply make use of frozen distributions
        if str(ptype) == 'uniform':
            priors_lst.append(ss.uniform(loc=pmean, scale=pstdd-pmean))

        elif str(ptype) == 'normal':
            priors_lst.append(ss.norm(loc=pmean, scale=pstdd))

        elif str(ptype) == 'gamma':
            b = pstdd**2/pmean
            a = pmean/b
            priors_lst.append(ss.gamma(a, scale=b))

        elif str(ptype) == 'beta':
            a = (1-pmean)*pmean**2/pstdd**2 - pmean
            b = a*(1/pmean - 1)
            priors_lst.append(ss.beta(a=a, b=b))

        elif str(ptype) == 'inv_gamma':

            def targf(x):
                y0 = ss.invgamma(x[0], scale=x[1]).std() - pstdd
                y1 = ss.invgamma(x[0], scale=x[1]).mean() - pmean
                return np.array([y0, y1])

            ig_res = so.root(targf, np.array([4, 4]))
            if ig_res['success']:
                a = ig_res['x']
                priors_lst.append(ss.invgamma(a[0], scale=a[1]))
            else:
                raise ValueError(
                    'Can not find inverse gamma distribution with mean %s and std %s' % (pmean, pstdd))
        elif str(ptype) == 'inv_gamma_dynare':
            s, nu = inv_gamma_spec(pmean, pstdd)
            ig = InvGammaDynare()(s, nu)
            # ig = ss.invgamma(nu/2, scale=s/2)
            priors_lst.append(ig)

        else:
            raise NotImplementedError(
                ' Distribution *not* implemented: ', str(ptype))
        if len(dist) == 3:
            print('  parameter %s as %s with mean %s and std/df %s...' %
                  (pp, ptype, pmean, pstdd))
        if len(dist) == 6:
            print('  parameter %s as %s (%s, %s). Init @ %s, with bounds (%s, %s)...' % (
                pp, ptype, pmean, pstdd, dist[0], dist[1], dist[2]))

    return priors_lst, initv, (lb, ub)


def pmdm_report(self, x_max, res_max, n=np.inf, printfunc=print):

    # getting the number of colums isn't that easy
    with os.popen('stty size', 'r') as rows_cols:
        cols = rows_cols.read().split()[1]

    if self.description is not None:
        printfunc('[bayesian_estimation -> pmdm ('+self.name+'):]'.ljust(45, ' ') +
                  ' Current best guess @ %s and ll of %s (%s):' % (n, -res_max.round(5), str(self.description)))
    else:
        printfunc('[bayesian_estimation -> pmdm ('+self.name+'):]'.ljust(45, ' ') +
                  ' Current best guess @ %s and ll of %s):' % (n, -res_max.round(5)))

    # split the info such that it is readable
    lnum = (len(self.priors)*8)//(int(cols)-8) + 1
    priors_chunks = np.array_split(np.array(self.fdict['prior_names']), lnum)
    vals_chunks = np.array_split([round(m_val, 3) for m_val in x_max], lnum)

    for pchunk, vchunk in zip(priors_chunks, vals_chunks):

        row_format = "{:>8}" * (len(pchunk) + 1)
        printfunc(row_format.format("", *pchunk))
        printfunc(row_format.format("", *vchunk))
        printfunc('')

    printfunc('')

    return
