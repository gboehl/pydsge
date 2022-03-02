#!/bin/python
# -*- coding: utf-8 -*-

import warnings
import os
import time
import tqdm
import numpy as np
import emcwrap as ew
import pandas as pd
import scipy.stats as ss
import scipy.optimize as so
from scipy.special import gammaln
from grgrlib import timeprint
from grgrlib.stats import mode

from emcwrap.dists import InvGammaDynare # to maintain picklability

def pmdm_report(self, x_max, res_max, n=np.inf, printfunc=print):

    # getting the number of colums isn't that easy
    with os.popen("stty size", "r") as rows_cols:
        cols = rows_cols.read().split()[1]

    if self.description is not None:
        printfunc(
            "[estimation -> pmdm ("
            + self.name
            + "):]".ljust(15, " ")
            + " Current best guess @ %s and ll of %s (%s):"
            % (n, -res_max.round(5), str(self.description))
        )
    else:
        printfunc(
            "[estimation -> pmdm ("
            + self.name
            + "):]".ljust(15, " ")
            + " Current best guess @ %s and ll of %s):" % (n, -res_max.round(5))
        )

    # split the info such that it is readable
    lnum = (len(self.prior) * 8) // (int(cols) - 8) + 1
    prior_chunks = np.array_split(np.array(self.fdict["prior_names"]), lnum)
    vals_chunks = np.array_split([round(m_val, 3) for m_val in x_max], lnum)

    for pchunk, vchunk in zip(prior_chunks, vals_chunks):

        row_format = "{:>8}" * (len(pchunk) + 1)
        printfunc(row_format.format("", *pchunk))
        printfunc(row_format.format("", *vchunk))
        printfunc("")

    printfunc("")

    return


def gfevd(
    self, eps_dict, horizon=1, nsamples=None, linear=False, seed=0, verbose=True, **args
):
    """Calculates the generalized forecasting error variance decomposition (GFEVD, Lanne & Nyberg)

    Parameters
    ----------
    eps : array or dict
    nsamples : int, optional
        Sample size. Defaults to everything exposed to the function.
    verbose : bool, optional
    """
    np.random.seed(seed)

    states = eps_dict["means"]
    if np.ndim(states) == 2:
        states = np.expand_dims(states, 0)
    states = states[:, :-1, :]

    pars = eps_dict["pars"]
    resids = eps_dict["resid"]

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

    for s in wrap(sample, total=nsamples, unit="draws", dynamic_ncols=True):

        if s[2] is not None:
            self.set_par(s[2], **args)

        for e in self.shocks:

            ei = self.shocks.index(e)
            shock = (e, s[0][ei], 0)

            irfs = self.irfs(shock, T=horizon, state=s[1], linear=linear)[0].to_numpy()[
                -1
            ]
            void = self.irfs((e, 0, 0), T=horizon, state=s[1], linear=linear)[
                0
            ].to_numpy()[-1]
            gis[ei] += (irfs - void) ** 2

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
        mbs += np.sum(vvd[ind[0]] ** 2) + \
            np.sum(vvd[:, ind[1]] ** 2) - vvd[ind] ** 2
        vvd = np.delete(vvd, ind[0], 0)
        vvd = np.delete(vvd, ind[1], 1)

    mbs /= 2 * (len(self.shocks) - 1)

    if verbose:
        print("[mbs_index:]".ljust(15, " ") +
              " MBS index is %s." % mbs.round(3))

    return mbs


def nhd(self, eps_dict, linear=False, **args):
    """Calculates the normalized historic decomposition, based on normalized counterfactuals"""

    states = eps_dict["init"]
    pars = eps_dict["pars"]
    resids = eps_dict["resid"]

    nsamples = pars.shape[0]
    hd = np.empty((nsamples, self.dimeps, self.data.shape[0], self.dimx))
    means = np.empty((nsamples, self.data.shape[0], self.dimx))
    rcons = np.empty(self.dimeps)

    # not paralellized
    for i in tqdm.tqdm(range(nsamples), unit=" sample(s)", dynamic_ncols=True):

        self.set_par(pars[i], **args)

        pmat, qmat, pterm, qterm, bmat, bterm = self.precalc_mat
        qmat = qmat[:, :, : -self.dimeps]
        qterm = qterm[..., : -self.dimeps]

        state = states[i]
        means[i, 0, :] = state

        hd[i, :, 0, :] = state / self.dimeps

        for t, resid in enumerate(resids[i]):
            state, (l, k), _ = self.t_func(
                state, resid, return_k=True, linear=linear)
            means[i, t + 1, :] = state

            # for each shock:
            for s in range(self.dimeps):

                eps = np.zeros(self.dimeps)
                eps[s] = resid[s]

                v = np.hstack((hd[i, s, t, -self.dimq + self.dimeps:], eps))
                p = pmat[l, k] @ v
                q = qmat[l, k] @ v
                hd[i, s, t + 1, :] = np.hstack((p, q))

                if k:
                    rcons[s] = bmat[0, l, k] @ v

            if k and rcons.sum():
                for s in range(len(self.shocks)):
                    # proportional to relative contribution to constaint spell duration
                    hd[i, s, t + 1, :] += (
                        rcons[s] / rcons.sum() *
                        np.hstack((pterm[l, k], qterm[l, k]))
                    )

    # as a list of DataFrames
    hd = [
        pd.DataFrame(h, index=self.data.index, columns=self.vv) for h in hd.mean(axis=0)
    ]
    means = pd.DataFrame(means.mean(
        axis=0), index=self.data.index, columns=self.vv)

    return hd, means


def mdd(
    self, method="laplace", chain=None, lprobs=None, tune=None, verbose=False, **args
):
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

    if method in ("laplace", "lp"):
        mstr = "LaPlace approximation"
        mdd = ew.mdd_laplace(chain, lprobs, calc_hess=False, **args)
    elif method == "hess":
        mstr = "LaPlace approximation with hessian approximation"
        mdd = ew.mdd_laplace(chain, lprobs, calc_hess=True, **args)
    elif method == "mhm":
        mstr = "modified harmonic mean"
        pool = self.pool if hasattr(self, "pool") else None
        mdd = ew.mdd_harmonic_mean(
            chain, lprobs, pool=pool, verbose=verbose > 1, **args)
    else:
        raise NotImplementedError(
            "[mdd:]".ljust(15, " ")
            + "`method` must be one of `laplace`, `mhm` or `hess`."
        )

    if verbose:
        print(
            "[mdd:]".ljust(15, " ")
            + "done after %s. Marginal data density according to %s is %s."
            % (timeprint(time.time() - st), mstr, mdd.round(3))
        )

    return mdd


def post_mean(self, chain=None, tune=None):
    """Calculate the mean of the posterior distribution"""

    tune = tune or self.get_tune
    chain = chain or self.get_chain()[-tune:]

    return chain.reshape(-1, chain.shape[-1]).mean(axis=0)


def sort_nhd(hd):
    """Sort the normalized historical decomposition into negative and positive contributions"""

    hmin = np.zeros_like(hd[0])
    hmax = np.zeros_like(hd[0])
    hmaxt = ()
    hmint = ()

    for h in hd:
        newmax = hmax + np.where(h > 0, h, 0)
        hmaxt += (np.stack((hmax, newmax)),)
        newmin = hmin + np.where(h < 0, h, 0)
        hmint += (np.stack((hmin, newmin)),)
        hmin = newmin
        hmax = newmax

    return hmint, hmaxt
