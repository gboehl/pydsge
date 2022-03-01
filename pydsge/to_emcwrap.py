#!/bin/python
# -*- coding: utf-8 -*-

import emcee
import tqdm
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.optimize as so
from scipy.special import gammaln
from grgrlib import map2arr
from grgrlib.stats import logpdf


def get_prior(prior, verbose=False):

    prior_lst = []
    initv, lb, ub = [], [], []

    if verbose:
        print("Adding parameters to the prior distribution...")

    for pp in prior:

        dist = prior[str(pp)]

        if len(dist) == 3:
            initv.append(None)
            lb.append(None)
            ub.append(None)
            ptype, pmean, pstdd = dist
        elif len(dist) == 6:
            initv.append(eval(str(dist[0])))
            lb.append(dist[1])
            ub.append(dist[2])
            ptype, pmean, pstdd = dist[3:]
        else:
            raise NotImplementedError(
                "Prior specification must either be 3 or 6 inputs but is %s." % pp)

        # simply make use of frozen distributions
        if str(ptype) == "uniform":
            prior_lst.append(ss.uniform(loc=pmean, scale=pstdd - pmean))

        elif str(ptype) == "normal":
            prior_lst.append(ss.norm(loc=pmean, scale=pstdd))

        elif str(ptype) == "gamma":
            b = pstdd ** 2 / pmean
            a = pmean / b
            prior_lst.append(ss.gamma(a, scale=b))

        elif str(ptype) == "beta":
            a = (1 - pmean) * pmean ** 2 / pstdd ** 2 - pmean
            b = a * (1 / pmean - 1)
            prior_lst.append(ss.beta(a=a, b=b))

        elif str(ptype) == "inv_gamma":

            def targf(x):
                y0 = ss.invgamma(x[0], scale=x[1]).std() - pstdd
                y1 = ss.invgamma(x[0], scale=x[1]).mean() - pmean
                return np.array([y0, y1])

            ig_res = so.root(targf, np.array([4, 4]), method="lm")

            if ig_res["success"] and np.allclose(targf(ig_res["x"]), 0):
                prior_lst.append(ss.invgamma(
                    ig_res["x"][0], scale=ig_res["x"][1]))
            else:
                raise ValueError(
                    f"Can not find inverse gamma distribution with mean {pmean} and std {pstdd}.")

        elif str(ptype) == "inv_gamma_dynare":
            s, nu = inv_gamma_spec(pmean, pstdd)
            ig = InvGammaDynare()(s, nu)
            prior_lst.append(ig)

        else:
            raise NotImplementedError(
                f" Distribution {ptype} not implemented.")
        if verbose:
            if len(dist) == 3:
                print(
                    "   - %s as %s with mean %s and std/df %s"
                    % (pp, ptype, pmean, pstdd)
                )
            if len(dist) == 6:
                print(
                    "   - %s as %s (%s, %s). Init @ %s, with bounds (%s, %s)"
                    % (pp, ptype, pmean, pstdd, dist[0], dist[1], dist[2])
                )

    def lprior(par):

        prior = 0
        for i, pl in enumerate(prior_lst):
            prior += pl.logpdf(par[i])

        return prior

    return prior_lst, lprior, initv, (lb, ub)


# to dists.py


class InvGammaDynare(ss.rv_continuous):

    name = "inv_gamma_dynare"

    def _logpdf(self, x, s, nu):

        if x < 0:
            lpdf = -np.inf
        else:
            lpdf = (
                np.log(2)
                - gammaln(nu / 2)
                - nu / 2 * (np.log(2) - np.log(s))
                - (nu + 1) * np.log(x)
                - 0.5 * s / np.square(x)
            )

        return lpdf

    def _pdf(self, x, s, nu):
        return np.exp(self._logpdf(x, s, nu))


def inv_gamma_spec(mu, sigma):

    # directly stolen and translated from dynare/matlab

    def ig1fun(nu):
        return (
            np.log(2 * mu ** 2)
            - np.log((sigma ** 2 + mu ** 2) * (nu - 2))
            + 2 * (gammaln(nu / 2) - gammaln((nu - 1) / 2))
        )

    nu = np.sqrt(2 * (2 + mu ** 2 / sigma ** 2))
    nu2 = 2 * nu
    nu1 = 2
    err = ig1fun(nu)
    err2 = ig1fun(nu2)

    if err2 > 0:
        while nu2 < 1e12:  # Shift the interval containing the root.
            nu1 = nu2
            nu2 = nu2 * 2
            err2 = ig1fun(nu2)
            if err2 < 0:
                break
        if err2 > 0:
            raise ValueError(
                "[inv_gamma_spec:] Failed in finding an interval containing a sign change! You should check that the prior variance is not too small compared to the prior mean..."
            )

    # Solve for nu using the secant method.
    while abs(nu2 / nu1 - 1) > 1e-14:
        if err > 0:
            nu1 = nu
            if nu < nu2:
                nu = nu2
            else:
                nu = 2 * nu
                nu2 = nu
        else:
            nu2 = nu
        nu = (nu1 + nu2) / 2
        err = ig1fun(nu)

    s = (sigma ** 2 + mu ** 2) * (nu - 2)

    if (
        abs(
            np.log(mu)
            - np.log(np.sqrt(s / 2))
            - gammaln((nu - 1) / 2)
            + gammaln(nu / 2)
        )
        > 1e-7
    ):
        raise ValueError(
            "[inv_gamma_spec:] Failed in solving for the hyperparameters!")
    if abs(sigma - np.sqrt(s / (nu - 2) - mu * mu)) > 1e-7:
        raise ValueError(
            "[inv_gamma_spec:] Failed in solving for the hyperparameters!")

    return s, nu


def summary(priors, store, pmode=None, bounds=None, alpha=0.1, top=None, show_prior=True):
    # inspired by pymc3 because it looks really nice

    if bounds is not None or isinstance(store, tuple):
        xs, fs, ns = store
        ns = ns.squeeze()
        fas = (-fs[:, 0]).argsort()
        xs = xs[fas]
        fs = fs.squeeze()[fas]

    f_prs = [
        lambda x: pd.Series(x, name="distribution"),
        lambda x: pd.Series(x, name="pst_mean"),
        lambda x: pd.Series(x, name="sd/df"),
    ]

    f_bnd = [
        lambda x: pd.Series(x, name="lbound"),
        lambda x: pd.Series(x, name="ubound"),
    ]

    def mode_func(x, n):
        return pmode[n] if pmode is not None else mode(x.flatten())

    funcs = [
        lambda x, n: pd.Series(np.mean(x), name="mean"),
        lambda x, n: pd.Series(np.std(x), name="sd"),
        lambda x, n: pd.Series(
            mode_func(x, n), name="mode" if pmode is not None else "marg. mode"
        ),
        lambda x, n: _hpd_df(x, alpha),
        lambda x, n: pd.Series(mc_error(x), name="error"),
    ]

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
            [lst.append(f("")) for j, f in enumerate(f_prs)]
            if bounds is not None:
                [lst.append(f("")) for j, f in enumerate(f_bnd)]

        [lst.append(pd.Series(s, name=n)) for s, n in zip(fs[:top], ns[:top])]
        var_df = pd.concat(lst, axis=1)
        var_df.index = ["loglike"]
        var_dfs.append(var_df)

    dforg = pd.concat(var_dfs, axis=0, sort=False)

    return dforg


def mcmc_summary(
    chain,
    lprobs,
    priors,
    acceptance_fraction=None,
    out=print,
    **args
):

    nchain = chain.reshape(-1, chain.shape[-1])
    lprobs = lprobs.reshape(-1, lprobs.shape[-1])
    mode_x = nchain[lprobs.argmax()]

    res = summary(priors, chain, mode_x, **args)

    out(res.round(3))

    if acceptance_fraction is not None:

        out("Mean acceptance fraction:" +
            str(np.mean(acceptance_fraction).round(3)).rjust(13))

    return res


def mcmc(lprob, p0, nsteps, moves=None, priors=None, backend=None, update_freq=None, resume=False, pool=None, report=None, description=None, temp=1, maintenance_interval=False, seed=None, verbose=False, **kwargs):

    nwalks, ndim = np.shape(p0)

    if seed is None:
        seed = 0

    np.random.seed(seed)

    if isinstance(backend, str):
        backend = emcee.backends.HDFBackend(
            os.path.splitext(backend)[0] + '.h5')

    if resume:
        nwalks = backend.get_chain().shape[1]

    if update_freq is None:
        update_freq = nsteps // 5

    sampler = emcee.EnsembleSampler(
        nwalks, ndim, lprob, moves=moves, pool=pool, backend=backend)

    if resume and not p0:
        p0 = sampler.get_last_sample()

    if not verbose:  # verbose means VERY verbose
        np.warnings.filterwarnings("ignore")

    if verbose > 2:
        report = report or print
    else:
        pbar = tqdm.tqdm(total=nsteps, unit="sample(s)", dynamic_ncols=True)
        report = report or pbar.write

    old_tau = np.inf
    cnt = 0

    for result in sampler.sample(p0, iterations=nsteps, **kwargs):

        if not verbose:
            lls = list(result)[1]
            maf = np.mean(sampler.acceptance_fraction[-update_freq:]) * 100
            pbar.set_description(
                "[ll/MAF:%s(%1.0e)/%1.0f%%]" % (str(np.max(lls))
                                                [:7], np.std(lls), maf)
            )

        if cnt and update_freq and not cnt % update_freq:

            prnttup = "(mcmc:) Summary from last %s of %s iterations" % (
                update_freq, cnt)

            if temp < 1:
                prnttup += " with temp of %s%%" % (np.round(temp * 100, 6))

            if description is not None:
                prnttup += " (%s)" % str(description)

            prnttup += ":"

            report(prnttup)

            sample = sampler.get_chain()
            lprobs = sampler.get_log_prob(flat=True)
            acfs = sampler.acceptance_fraction

            tau = emcee.autocorr.integrated_time(sample, tol=0)
            min_tau = np.min(tau).round(2)
            max_tau = np.max(tau).round(2)
            dev_tau = np.max(np.abs(old_tau - tau) / tau)

            tau_sign = ">" if max_tau > sampler.iteration / 50 else "<"
            dev_sign = ">" if dev_tau > 0.01 else "<"

            if priors is not None:
                mcmc_summary(
                    chain=sample[-update_freq:],
                    lprobs=lprobs[-update_freq:],
                    priors=priors,
                    acceptance_fraction=acfs[-update_freq:],
                    out=lambda x: report(str(x)),
                )

            report(
                "Convergence stats: tau is in (%s,%s) (%s%s) and change is %s (%s0.01)."
                % (
                    min_tau,
                    max_tau,
                    tau_sign,
                    sampler.iteration / 50,
                    dev_tau.round(3),
                    dev_sign,
                )
            )

        if cnt and update_freq and not (cnt + 1) % update_freq:
            sample = sampler.get_chain()
            old_tau = emcee.autocorr.integrated_time(sample, tol=0)

        if not verbose:
            pbar.update(1)

        # avoid mem leakage
        if maintenance_interval and cnt and pool and not cnt % maintenance_interval:
            pool.clear()

        cnt += 1

    pbar.close()
    if pool:
        pool.close()

    if not verbose:
        np.warnings.filterwarnings("default")

    if backend is None:
        return sampler
    else:
        return backend


def _hpd_df(x, alpha):

    cnames = [
        "hpd_{0:g}".format(100 * alpha / 2),
        "hpd_{0:g}".format(100 * (1 - alpha / 2)),
    ]

    sx = np.sort(x.flatten())
    hpd_vals = np.array(calc_min_interval(sx, alpha)).reshape(1, -1)

    return pd.DataFrame(hpd_vals, columns=cnames)


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
        warnings.warn("Too few elements for interval calculation.")

        return None, None

    else:
        min_idx = np.argmin(interval_width)
        hdi_min = x[min_idx]
        hdi_max = x[min_idx + interval_idx_inc]

        return hdi_min, hdi_max


def mc_error(x):
    means = np.mean(x, 0)
    return np.std(means) / np.sqrt(x.shape[0])


def get_prior_sample(frozen_prior, nsamples, check_func=False, seed=None, mapper=map, verbose=True):

    if seed is None:
        seed = 0

    if check_func and not callable(check_func):
        raise Exception('`check_func` must be `False` or a callable')

    def runner(locseed):

        np.random.seed(seed + locseed)
        done = False
        no = 0

        while not done:

            no += 1

            with np.warnings.catch_warnings(record=False):
                try:
                    np.warnings.filterwarnings("error")
                    rst = np.random.randint(2 ** 31)  # win explodes with 2**32
                    pdraw = [
                        pl.rvs(random_state=rst + sn)
                        for sn, pl in enumerate(frozen_prior)
                    ]

                    if check_func:
                        draw_prob = check_func(pdraw)
                        done = not np.any(np.isinf(draw_prob))

                except Exception as e:
                    if verbose > 1:
                        print(str(e) + " (%s) " % no)
                    if not locseed and no == 10:
                        raise

        return pdraw, no

    if verbose > 1:
        print("[prior_sample:]".ljust(15, " ") + " Sampling from the pior...")

    wrapper = tqdm.tqdm if verbose < 2 else (lambda x, **kwarg: x)
    pmap_sim = wrapper(mapper(runner, range(nsamples)), total=nsamples)

    draws, nos = map2arr(pmap_sim)

    if verbose and check_func:
        print(
            "[prior_sample:]".ljust(15, " ")
            + " Sampling done. Check fails for %2.2f%% of the prior."
            % (100 * (sum(nos) - nsamples) / sum(nos))
        )

    return draws


def mdd_laplace(chain, lprobs, calc_hess=False):
    """Approximate the marginal data density useing the LaPlace method."""

    if chain.ndim > 2:
        chain = chain.reshape(-1, chain.shape[2])

    lprobs = lprobs.flatten()

    mode_x = chain[lprobs.argmax()]

    if calc_hess:

        import numdifftools as nd

        np.warnings.filterwarnings("ignore")
        hh = nd.Hessian(func)(mode_x)
        np.warnings.filterwarnings("default")

        if np.isnan(hh).any():
            raise ValueError(
                "[mdd:]".ljust(15, " ")
                + "Option `hess` is experimental and did not return a usable hessian matrix."
            )

        inv_hess = np.linalg.inv(hh)

    else:
        inv_hess = np.cov(chain.T)

    ndim = chain.shape[-1]
    log_det_inv_hess = np.log(np.linalg.det(inv_hess))
    mdd = 0.5 * ndim * np.log(2 * np.pi) + 0.5 * \
        log_det_inv_hess + lprobs.max()

    return mdd


def mdd_harmonic_mean(chain, lprobs, pool=None, alpha=0.05, verbose=False, debug=False):
    """Approximate the marginal data density useing modified harmonic mean."""

    if chain.ndim > 2:
        chain = chain.reshape(-1, chain.shape[2])

    lprobs = lprobs.flatten()

    cmean = chain.mean(axis=0)
    ccov = np.cov(chain.T)
    cicov = np.linalg.inv(ccov)

    nsamples = chain.shape[0]

    def runner(chunk):

        res = np.empty_like(chunk)
        wrapper = tqdm.tqdm if verbose else (lambda x, **kwarg: x)

        for i in wrapper(range(len(chunk))):

            drv = chain[i]
            drl = lprobs[i]

            if (drv - cmean) @ cicov @ (drv - cmean) < ss.chi2.ppf(
                1 - alpha, df=chain.shape[-1]
            ):
                res[i] = logpdf(drv, cmean, ccov) - drl
            else:
                res[i] = -np.inf

        return res

    if not debug and pool is not None:
        nbatches = pool.ncpus
        batches = pool.imap(runner, np.array_split(chain, nbatches))
        mls = np.vstack(list(batches))
    else:
        mls = runner(chain)

    maxllike = np.max(mls)  # for numeric stability
    imdd = np.log(np.mean(np.exp(mls - maxllike))) + maxllike

    return -imdd
