#!/bin/python
# -*- coding: utf-8 -*-

import emcee
import tqdm
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.optimize as so
from scipy.special import gammaln


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

    return prior_lst, initv, (lb, ub)


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
        lambda x, n: pd.Series(mc_error(x), name="mc_error"),
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


def mcmc(lprob, p0, nwalks, nsteps, moves, tune, priors, backend=None, update_freq=None, resume=False, pool=None, report=None, description=None, temp=1, maintenance_interval=10, debug=False, verbose=True, **kwargs):

    ndim = len(priors)

    if resume:
        nwalks = backend.get_chain().shape[1]

    if update_freq is None:
        update_freq = nsteps // 5

    if debug:
        sampler = emcee.EnsembleSampler(nwalks, ndim, lprob)
    else:
        sampler = emcee.EnsembleSampler(
            nwalks, ndim, lprob, moves=moves, pool=pool, backend=backend
        )

    if resume and not p0:
        p0 = sampler.get_last_sample()

    if not verbose:
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

            prnttup = "[mcmc:]".ljust(
                15, " "
            ) + "Summary from last %s of %s iterations" % (update_freq, cnt)

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
        if cnt and not cnt % maintenance_interval:
            pool.clear()

        cnt += 1

    pbar.close()
    if pool:
        pool.close()

    if not verbose:
        np.warnings.filterwarnings("default")

    log_probs = sampler.get_log_prob()[-tune:]
    chain = sampler.get_chain()[-tune:]
    chain = chain.reshape(-1, chain.shape[-1])

    return sampler, chain, log_probs


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
