#!/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from .stats import gfevd, mbcs_index, nhd, mdd
from .mcmc import mcmc
from .filtering import *
from .tools import *
from .mpile import *
from .estimation import *
import emcwrap as ew


class DSGE_RAW(dict):
    pass


def vix(self, variables, dontfail=False):
    """Returns the indices of a list of variables"""

    if isinstance(variables, str):
        variables = [variables]

    res = []
    for v in variables:
        try:
            res.append(list(self.vv).index(v))
        except ValueError:
            if not dontfail:
                raise

    return res


def oix(self, observables):
    """Returns the indices of a list of observables"""

    if isinstance(observables, str):
        observables = [observables]

    return [list(self.observables).index(v) for v in observables]


@property
def get_tune(self):

    if hasattr(self, "tune"):
        return self.tune
    else:
        return self.fdict["tune"]


def get_chain(
    self,
    get_acceptance_fraction=False,
    get_log_prob=False,
    backend_file=None,
    flat=None,
):

    if not backend_file:
        if hasattr(self, "sampler"):
            reader = self.sampler
        elif "backend_file" in self.fdict.keys():
            backend_file = str(self.fdict["backend_file"])
        else:
            backend_file = os.path.join(self.path, self.name + "_sampler.h5")

    if backend_file:
        if not os.path.exists(backend_file):
            raise NameError(
                "A backend file named `%s` could not be found." % backend_file
            )

        import emcee

        reader = emcee.backends.HDFBackend(backend_file, read_only=True)

    if get_acceptance_fraction:
        try:
            return reader.acceptance_fraction
        except:
            return reader.accepted / reader.iteration

    if get_log_prob:
        return reader.get_log_prob(flat=flat)

    chain = reader.get_chain(flat=flat)

    # ensure that bptrans exists and is applied
    if not hasattr(self, 'bptrans'):
        self.load_estim()

    if self.bptrans:
        return self.bptrans(chain)

    return chain


def get_log_prob(self, **args):
    """Get the log likelihoods in the chain"""
    # just a wrapper
    return get_chain(self, get_log_prob=True, **args)


def write_yaml(self, filename):

    if filename[-5:] != ".yaml":
        filename = filename + ".yaml"

    f = open(filename, "w+")

    f.write(self.raw_yaml)
    f.close()

    print("Model written to '%s.'" % filename)

    return


def save_meta(self, filename=None, verbose=True):

    import os

    filename = filename or os.path.join(self.path, self.name + "_meta")

    objs = "description", "backend_file", "tune", "name"

    for o in objs:
        if hasattr(self, o):
            exec("self.fdict[o] = self." + str(o))

    if hasattr(self, "filter"):
        try:
            self.fdict["filter_R"] = self.filter.R
            self.fdict["filter_P"] = self.filter.P
        except:
            pass

    np.savez_compressed(filename, **self.fdict)

    if verbose:
        print("[save_meta:]".ljust(15, " ") +
              " Metadata saved as '%s'" % filename)

    return


def save_rdict(self, rdict, path=None, suffix="", verbose=True):
    """Save dictionary of results

    The idea is to keep meta data (the model, setup, ...) and the results obtained (chains, smoothed residuals, ...) separate.
    """

    if not isinstance(path, str):
        path = self.name + "_res"

    if path[-4:] == ".npz":
        path = path[:-4]

    if not os.path.isabs(path):
        path = os.path.join(self.path, path)

    np.savez_compressed(path + suffix, **rdict)

    if verbose:
        print(
            "[save_rdict:]".ljust(15, " ") +
            " Results saved as '%s'" % (path + suffix)
        )
    return


def load_rdict(self, path=None, suffix=""):
    """Load stored dictionary of results

    The idea is to keep meta data (the model, setup, ...) and the results obtained (chains, smoothed residuals, ...) separate. `save_rdict` suggests some standard conventions.
    """

    if path is None:
        path = self.name + "_res"

    path += suffix

    if path[-4] != ".npz":
        path += ".npz"

    if not os.path.isabs(path):
        path = os.path.join(self.path, path)

    return dict(np.load(path, allow_pickle=True))


def traceplot_m(self, chain=None, prior_names=None, **args):

    if chain is None:
        chain = self.get_chain()
        args["tune"] = self.get_tune

    if prior_names is None:
        prior_names = self.fdict["prior_names"]

    return ew.traceplot(chain, varnames=prior_names, **args)


def posteriorplot_m(self, **args):

    tune = self.get_tune

    return ew.posteriorplot(
        self.get_chain(), varnames=self.fdict["prior_names"], tune=tune, **args
    )


def mcmc_summary(
    self,
    chain=None,
    tune=None,
    **args
):

    try:
        chain = self.get_chain() if chain is None else chain
    except AttributeError:
        raise AttributeError("[summary:]".ljust(
            15, " ") + "No chain to be found...")

    tune = tune or self.get_tune
    lprobs = self.get_log_prob()

    return ew.mcmc_summary(chain[-tune:], lprobs[-tune:], priors=self.prior, acceptance_fraction=self.get_chain(get_acceptance_fraction=True), **args)


def posterior2csv(self, path=None, tune=None, **args):

    tune = tune or self.get_tune
    path = path or os.path.join(self.path, self.name + "_posterior.csv")

    chain = self.get_chain()
    post = chain[-tune:].reshape(-1, chain.shape[-1])

    vd = pd.DataFrame(post.T, index=self.prior_names)
    vd.to_csv(path)

    return


def info_m(self, verbose=True, **args):

    try:
        name = self.name
    except AttributeError:
        name = self.fdict["name"]

    try:
        description = self.description
    except AttributeError:
        description = self.fdict["description"]

    try:
        dtime = str(self.fdict["datetime"])
    except KeyError:
        dtime = ""

    res = "Title: %s\n" % name
    res += "Date: %s\n" % dtime if dtime else ""
    res += "Description: %s\n" % description

    try:
        cshp = self.get_chain().shape
        tune = self.get_tune
        res += "Parameters: %s\n" % cshp[2]
        res += "Chains: %s\n" % cshp[1]
        res += "Last %s of %s samples\n" % (tune, cshp[0])
    except (AttributeError, KeyError):
        pass

    if verbose:
        print(res)

    return res


def load_data(self, df, start=None, end=None):
    """Load and prepare data
    ...
    This function takes a provided `pandas.DataFrame`, reads out the observables as they are defined in the YAML-file, and ajusts it regarding the `start` and `end` keywords. Using a `pandas.DatetimeIndex` as index of the DataFrame is strongly encuraged as it can be very powerful, but not necessary.

    Parameters
    ----------
    df : pandas.DataFrame
    start : index (optional)
    end : index (optional)

    Returns
    -------
    pandas.DataFrame

    """

    import cloudpickle as cpickle

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Type of input data must be a `pandas.DataFrame`.")

    if self is not None:
        for o in self.observables:
            if str(o) not in df.keys():
                raise KeyError("%s is not in the data!" % o)

        d = df[self.observables]

    if start is not None:
        start = str(start)

    if end is not None:
        end = str(end)

    d = d.loc[start:end]

    if np.any(d.isna()):
        raise Exception("Data must not contain `NaN`s.")

    self.data = d
    self.fdict["data"] = cpickle.dumps(d, protocol=4)
    self.fdict["obs"] = self.observables

    return d


def get_sample(self, size, chain=None):
    """Get a (preferably recent) sample from the chain"""

    chain = None or self.get_chain()
    clen, nwalks, npar = chain.shape
    recent = int(np.ceil(60 / nwalks))

    if recent > clen:
        raise Exception("Requested sample size is larger than chain")

    sample = chain[:, -recent, :].reshape(-1, npar)
    res = np.random.choice(np.arange(recent * nwalks), size, False)

    return sample[res]


DSGE_RAW.vix = vix
DSGE_RAW.oix = oix
DSGE_RAW.get_tune = get_tune
DSGE_RAW.save = save_meta
DSGE_RAW.mapper = mapper
DSGE_RAW.mcmc_summary = mcmc_summary
DSGE_RAW.info = info_m
DSGE_RAW.mdd = mdd
DSGE_RAW.get_data = load_data
DSGE_RAW.load_data = load_data
DSGE_RAW.get_sample = get_sample
DSGE_RAW.create_pool = create_pool
DSGE_RAW.posterior2csv = posterior2csv
# from mpile
DSGE_RAW.prior_sampler = prior_sampler
DSGE_RAW.get_par = get_par
DSGE_RAW.gp = get_par
DSGE_RAW.get_cov = get_cov
DSGE_RAW.set_par = set_par
DSGE_RAW.box_check = box_check
# from tools
DSGE_RAW.t_func = t_func
DSGE_RAW.o_func = o_func
DSGE_RAW.irfs = irfs
DSGE_RAW.simulate = simulate
DSGE_RAW.shock2state = shock2state
DSGE_RAW.obs = o_func
DSGE_RAW.get_eps_lin = get_eps_lin
DSGE_RAW.k_map = k_map
DSGE_RAW.traj = traj
# from mcmc
DSGE_RAW.mcmc = mcmc
# from estimation
DSGE_RAW.prep_estim = prep_estim
DSGE_RAW.load_estim = prep_estim
# DSGE_RAW.lprob = lprob
# from filter
DSGE_RAW.create_filter = create_filter
DSGE_RAW.get_p_init_lyapunov = get_p_init_lyapunov
DSGE_RAW.run_filter = run_filter
DSGE_RAW.get_ll = get_ll
# from plot
DSGE_RAW.traceplot = traceplot_m
DSGE_RAW.posteriorplot = posteriorplot_m
# from misc
DSGE_RAW.get_chain = get_chain
DSGE_RAW.get_log_prob = get_log_prob
DSGE_RAW.extract = extract
DSGE_RAW.create_obs_cov = create_obs_cov
DSGE_RAW.mask = mask
DSGE_RAW.load_rdict = load_rdict
DSGE_RAW.save_rdict = save_rdict
DSGE_RAW.gfevd = gfevd
DSGE_RAW.mbcs_index = mbcs_index
DSGE_RAW.nhd = nhd
