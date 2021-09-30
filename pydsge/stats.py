#!/bin/python
# -*- coding: utf-8 -*-

import warnings
import os
import time
import tqdm
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.optimize as so
from scipy.special import gammaln
from grgrlib.core import timeprint
from grgrlib.stats import mode


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


def summary(self, store, pmode=None, bounds=None, alpha=0.1, top=None, show_prior=True):
    # inspired by pymc3 because it looks really nice

    priors = self['__data__']['estimation']['prior']

    if bounds is not None or isinstance(store, tuple):
        xs, fs, ns = store
        ns = ns.squeeze()
        fas = (-fs[:, 0]).argsort()
        xs = xs[fas]
        fs = fs.squeeze()[fas]

    f_prs = [lambda x: pd.Series(x, name='distribution'),
             lambda x: pd.Series(x, name='pst_mean'),
             lambda x: pd.Series(x, name='sd/df')]

    f_bnd = [lambda x: pd.Series(x, name='lbound'),
             lambda x: pd.Series(x, name='ubound')]

    def mode_func(x, n): return pmode[n] if pmode is not None else mode(
        x.flatten())

    funcs = [
        lambda x, n: pd.Series(np.mean(x), name='mean'),
        lambda x, n: pd.Series(np.std(x), name='sd'),
        lambda x, n: pd.Series(
            mode_func(x, n), name='mode' if pmode is not None else 'marg. mode'),
        lambda x, n: _hpd_df(x, alpha),
        lambda x, n: pd.Series(mc_error(x), name='mc_error')]

    var_dfs = []
    for i, var in enumerate(priors):

        lst = []
        if show_prior:
            prior = priors[var]
            if len(prior) > 3:
                prior = prior[-3:]
            [lst.append(f(prior[j])) for j, f in enumerate(f_prs)]
            if bounds is not None:
                [lst.append(f(np.array(bounds).T[i][j]))
                 for j, f in enumerate(f_bnd)]

        if bounds is not None:
            [lst.append(pd.Series(s[i], name=n))
             for s, n in zip(xs[:top], ns[:top])]
        else:
            vals = store[:, :, i]
            [lst.append(f(vals, i)) for f in funcs]
        var_df = pd.concat(lst, axis=1)
        var_df.index = [var]
        var_dfs.append(var_df)

    if bounds is not None:

        lst = []

        if show_prior:
            [lst.append(f('')) for j, f in enumerate(f_prs)]
            if bounds is not None:
                [lst.append(f('')) for j, f in enumerate(f_bnd)]

        [lst.append(pd.Series(s, name=n)) for s, n in zip(fs[:top], ns[:top])]
        var_df = pd.concat(lst, axis=1)
        var_df.index = ['loglike']
        var_dfs.append(var_df)

    dforg = pd.concat(var_dfs, axis=0, sort=False)

    return dforg


def mc_mean(trace, varnames):
    # in most parts just stolen from pymc3 because it looks really nice

    p_means = []

    for i, var in enumerate(varnames):
        vals = trace[:, :, i]
        p_means.append(np.mean(vals))

    return p_means


class InvGammaDynare(ss.rv_continuous):

    name = 'inv_gamma_dynare'

    def _logpdf(self, x, s, nu):

        if x < 0:
            lpdf = -np.inf
        else:
            lpdf = np.log(2) - gammaln(nu/2) - nu/2*(np.log(2) -
                                                     np.log(s)) - (nu+1)*np.log(x) - .5*s/np.square(x)

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


def get_prior(prior, verbose=False):

    prior_lst = []
    initv = []
    lb = []
    ub = []

    if verbose:
        print('Adding parameters to the prior distribution...')

    for pp in prior:

        dist = prior[str(pp)]

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
                'Shape of prior specification of %s is unclear (!=3 & !=6).' % pp)

        # simply make use of frozen distributions
        if str(ptype) == 'uniform':
            prior_lst.append(ss.uniform(loc=pmean, scale=pstdd-pmean))

        elif str(ptype) == 'normal':
            prior_lst.append(ss.norm(loc=pmean, scale=pstdd))

        elif str(ptype) == 'gamma':
            b = pstdd**2/pmean
            a = pmean/b
            prior_lst.append(ss.gamma(a, scale=b))

        elif str(ptype) == 'beta':
            a = (1-pmean)*pmean**2/pstdd**2 - pmean
            b = a*(1/pmean - 1)
            prior_lst.append(ss.beta(a=a, b=b))

        elif str(ptype) == 'inv_gamma':

            def targf(x):
                y0 = ss.invgamma(x[0], scale=x[1]).std() - pstdd
                y1 = ss.invgamma(x[0], scale=x[1]).mean() - pmean
                return np.array([y0, y1])

            ig_res = so.root(targf, np.array([4, 4]), method='lm')

            if ig_res['success'] and np.allclose(targf(ig_res['x']), 0):
                prior_lst.append(ss.invgamma(
                    ig_res['x'][0], scale=ig_res['x'][1]))
            else:
                raise ValueError(
                    'Can not find inverse gamma distribution with mean %s and std %s' % (pmean, pstdd))

        elif str(ptype) == 'inv_gamma_dynare':
            s, nu = inv_gamma_spec(pmean, pstdd)
            ig = InvGammaDynare()(s, nu)
            prior_lst.append(ig)

        else:
            raise NotImplementedError(
                ' Distribution *not* implemented: ', str(ptype))
        if verbose:
            if len(dist) == 3:
                print('  parameter %s as %s with mean %s and std/df %s...' %
                      (pp, ptype, pmean, pstdd))
            if len(dist) == 6:
                print('  parameter %s as %s (%s, %s). Init @ %s, with bounds (%s, %s)...' % (
                    pp, ptype, pmean, pstdd, dist[0], dist[1], dist[2]))

    return prior_lst, initv, (lb, ub)


def pmdm_report(self, x_max, res_max, n=np.inf, printfunc=print):

    # getting the number of colums isn't that easy
    with os.popen('stty size', 'r') as rows_cols:
        cols = rows_cols.read().split()[1]

    if self.description is not None:
        printfunc('[estimation -> pmdm ('+self.name+'):]'.ljust(15, ' ') +
                  ' Current best guess @ %s and ll of %s (%s):' % (n, -res_max.round(5), str(self.description)))
    else:
        printfunc('[estimation -> pmdm ('+self.name+'):]'.ljust(15, ' ') +
                  ' Current best guess @ %s and ll of %s):' % (n, -res_max.round(5)))

    # split the info such that it is readable
    lnum = (len(self.prior)*8)//(int(cols)-8) + 1
    prior_chunks = np.array_split(np.array(self.fdict['prior_names']), lnum)
    vals_chunks = np.array_split([round(m_val, 3) for m_val in x_max], lnum)

    for pchunk, vchunk in zip(prior_chunks, vals_chunks):

        row_format = "{:>8}" * (len(pchunk) + 1)
        printfunc(row_format.format("", *pchunk))
        printfunc(row_format.format("", *vchunk))
        printfunc('')

    printfunc('')

    return


def gfevd(self, eps_dict, horizon=1, nsamples=None, linear=False, seed=0, verbose=True, **args):
    """Calculates the generalized forecasting error variance decomposition (GFEVD, Lanne & Nyberg)

    Parameters
    ----------
    eps : array or dict
    nsamples : int, optional
        Sample size. Defaults to everything exposed to the function.
    verbose : bool, optional
    """
    np.random.seed(seed)

    states = eps_dict['means']
    if np.ndim(states) == 2:
        states = np.expand_dims(states, 0)
    states = states[:, :-1, :]

    pars = eps_dict['pars']
    resids = eps_dict['resid']

    if np.ndim(resids) > 2:
        pars = pars.repeat(resids.shape[1], axis=0)
    if np.ndim(resids) > 2:
        resids = resids.reshape(-1, resids.shape[-1])
    if np.ndim(states) > 2:
        states = states.reshape(-1, states.shape[-1])

    nsamples = nsamples or resids.shape[0]
    numbers = np.arange(resids.shape[0])
    draw = np.random.choice(numbers, nsamples, replace=False)

    sample = zip(resids[draw], states[draw], pars[draw])
    gis = np.zeros((len(self.shocks), len(self.vv)))

    wrap = tqdm.tqdm if verbose else (lambda x, **kwarg: x)

    for s in wrap(sample, total=nsamples, unit='draws', dynamic_ncols=True):

        if s[2] is not None:
            self.set_par(s[2], **args)

        for e in self.shocks:

            ei = self.shocks.index(e)
            shock = (e, s[0][ei], 0)

            irfs = self.irfs(shock, T=horizon, state=s[1], linear=linear)[
                0].to_numpy()[-1]
            void = self.irfs((e, 0, 0), T=horizon, state=s[1], linear=linear)[
                0].to_numpy()[-1]
            gis[ei] += (irfs - void)**2

    gis /= np.sum(gis, axis=0)

    vd = pd.DataFrame(gis, index=self.shocks, columns=self.vv)

    if verbose > 1:
        print(vd.round(3))

    return vd


def mbcs_index(self, vd, verbose=True):
    """This implements a main-business-cycle shock measure

    Between 0 and 1, this indicates how well one single shock explains the business cycle dynamics
    """

    vvd = self.hx[0] @ vd.to_numpy().T

    mbs = 0
    for i in range(vvd.shape[0]):
        ind = np.unravel_index(vvd.argmax(), vvd.shape)
        vvd[ind] -= 1
        mbs += np.sum(vvd[ind[0]]**2) + np.sum(vvd[:, ind[1]]**2) - vvd[ind]**2
        vvd = np.delete(vvd, ind[0], 0)
        vvd = np.delete(vvd, ind[1], 1)

    mbs /= 2*(len(self.shocks) - 1)

    if verbose:
        print('[mbs_index:]'.ljust(15, ' ') +
              " MBS index is %s." % mbs.round(3))

    return mbs


def nhd(self, eps_dict, linear=False, **args):
    """Calculates the normalized historic decomposition, based on normalized counterfactuals
    """

    states = eps_dict['init']
    pars = eps_dict['pars']
    resids = eps_dict['resid']

    nsamples = pars.shape[0]
    hd = np.empty((nsamples, self.dimeps, self.data.shape[0], self.dimx))
    means = np.empty((nsamples, self.data.shape[0], self.dimx))
    rcons = np.empty(self.dimeps)

    # not paralellized
    for i in tqdm.tqdm(range(nsamples), unit=' sample(s)', dynamic_ncols=True):

        self.set_par(pars[i], **args)

        pmat, qmat, pterm, qterm, bmat, bterm = self.precalc_mat
        qmat = qmat[:, :, :-self.dimeps]
        qterm = qterm[..., :-self.dimeps]

        state = states[i]
        means[i, 0, :] = state

        hd[i, :, 0, :] = state/self.dimeps

        for t, resid in enumerate(resids[i]):
            state, (l, k), _ = self.t_func(
                state, resid, return_k=True, linear=linear)
            means[i, t+1, :] = state

            # for each shock:
            for s in range(self.dimeps):

                eps = np.zeros(self.dimeps)
                eps[s] = resid[s]

                v = np.hstack((hd[i, s, t, -self.dimq+self.dimeps:], eps))
                p = pmat[l, k] @ v
                q = qmat[l, k] @ v
                hd[i, s, t+1, :] = np.hstack((p, q))

                if k:
                    rcons[s] = bmat[0, l, k] @ v

            if k and rcons.sum():
                for s in range(len(self.shocks)):
                    # proportional to relative contribution to constaint spell duration
                    hd[i, s, t+1, :] += rcons[s] / \
                        rcons.sum()*np.hstack((pterm[l, k], qterm[l, k]))

    # as a list of DataFrames
    hd = [pd.DataFrame(h, index=self.data.index, columns=self.vv)
          for h in hd.mean(axis=0)]
    means = pd.DataFrame(means.mean(
        axis=0), index=self.data.index, columns=self.vv)

    return hd, means


def mdd_lp(chain, lprobs, calc_hess=False):
    """Approximate the marginal data density useing the LaPlace method.
    """

    mode_x = chain[lprobs.argmax()]

    if calc_hess:

        import numdifftools as nd

        np.warnings.filterwarnings('ignore')
        hh = nd.Hessian(func)(mode_x)
        np.warnings.filterwarnings('default')

        if np.isnan(hh).any():
            raise ValueError('[mdd:]'.ljust(
                15, ' ') + "Option `hess` is experimental and did not return a usable hessian matrix.")

        inv_hess = np.linalg.inv(hh)

    else:
        inv_hess = np.cov(chain.T)

    ndim = chain.shape[-1]
    log_det_inv_hess = np.log(np.linalg.det(inv_hess))
    mdd = .5*ndim*np.log(2*np.pi) + .5*log_det_inv_hess + lprobs.max()

    return mdd


def mdd_mhm(chain, lprobs, alpha=.05, pool=None, verbose=False, debug=False):
    """Approximate the marginal data density useing modified harmonic mean.
    """

    from grgrlib.stats import logpdf

    cmean = chain.mean(axis=0)
    ccov = np.cov(chain.T)
    cicov = np.linalg.inv(ccov)

    nsamples = chain.shape[0]
    ##

    def runner(chunk):
        ##
        res = np.empty_like(chunk)
        wrapper = tqdm.tqdm if verbose else (lambda x, **kwarg: x)
        ##
        for i in wrapper(range(len(chunk))):
            drv = chain[i]
            drl = lprobs[i]
            ##
            if (drv - cmean) @ cicov @ (drv - cmean) < ss.chi2.ppf(1-alpha, df=chain.shape[-1]):
                res[i] = logpdf(drv, cmean, ccov) - drl
            else:
                res[i] = -np.inf
        return res

    if pool and not debug:
        if pool.ncpus is None:
            import pathos
            nbatches = pathos.multiprocessing.cpu_count()
        else:
            nbatches = pool.ncpus

        batches = pool.imap(runner, np.array_split(chain, nbatches))
        mls = np.vstack(list(batches))
    else:
        mls = runner(chain)

    maxllike = np.max(mls)  # for numeric stability
    imdd = np.log(np.mean(np.exp(mls-maxllike))) + maxllike

    return -imdd


def mdd(self, method='laplace', chain=None, lprobs=None, tune=None, verbose=False, **args):
    """Approximate the marginal data density.

    Parameters
    ----------
    method : str
        The method used for the approximation. Can be either of 'laplace', 'mhm' (modified harmonic mean) or 'hess' (LaPlace approximation with the numerical approximation of the hessian; NOT FUNCTIONAL).
    """

    if verbose:
        st = time.time()

    if chain is None:
        tune = tune or self.get_tune
        chain = self.get_chain()[-tune:]
        chain = chain.reshape(-1, chain.shape[-1])

    if lprobs is None:
        tune = tune or self.get_tune
        lprobs = self.get_log_prob()[-tune:]
        lprobs = lprobs.flatten()

    if method in ('laplace', 'lp'):
        mstr = 'LaPlace approximation'
        mdd = mdd_lp(chain, lprobs, calc_hess=False, **args)
    elif method == 'hess':
        mstr = 'LaPlace approximation with hessian approximation'
        mdd = mdd_lp(chain, lprobs, calc_hess=True, **args)
    elif method == 'mhm':
        mstr = 'modified harmonic mean'
        pool = self.pool if hasattr(self, 'pool') else None
        mdd = mdd_mhm(chain, lprobs, pool=pool, verbose=verbose > 1, **args)
    else:
        raise NotImplementedError('[mdd:]'.ljust(
            15, ' ') + "`method` must be one of `laplace`, `mhm` or `hess`.")

    if verbose:
        print('[mdd:]'.ljust(15, ' ') + "done after %s. Marginal data density according to %s is %s." %
              (timeprint(time.time()-st), mstr, mdd.round(3)))

    return mdd


def post_mean(self, chain=None, tune=None):
    """Calculate the mean of the posterior distribution
    """

    tune = tune or self.get_tune
    chain = chain or self.get_chain()[-tune:]

    return chain.reshape(-1, chain.shape[-1]).mean(axis=0)
