#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
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
