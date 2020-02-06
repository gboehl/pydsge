#!/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from .parser import DSGE
from .stats import summary, gfevd, mbcs_index, nhd, mdd
from .plots import posteriorplot, traceplot
from .mcmc import mcmc, tmcmc
from .modesearch import cmaes
from .filtering import *
from .tools import *
from .core import get_sys, get_par, set_par
from .estimation import *


def vix(self, *variables):
    """Returns the indices of a list of variables
    """

    if isinstance(variables, str):
        variables = [variables]
    if len(variables) == 1:
        variables = variables[0]

    return [list(self.vv).index(v) for v in variables]


def get_eps_lin(self, x, xp, rcond=1e-14):
    """Get filter-implied (smoothed) shocks for linear model
    """

    return np.linalg.pinv(self.SIG, rcond) @ (x - self.lin_t_func@xp)


@property
def get_tune(self):

    if hasattr(self, 'tune'):
        return self.tune
    else:
        return self.fdict['tune']


def get_chain(self, get_acceptance_fraction=False, get_log_prob=False, backend_file=None, flat=None):

    if not backend_file:
        if hasattr(self, 'sampler'):
            reader = self.sampler
        elif 'backend_file' in self.fdict.keys():
            backend_file = str(self.fdict['backend_file'])
        else:
            backend_file = os.path.join(self.path, self.name+'_sampler.h5')

    if backend_file:

        if not os.path.exists(backend_file):
            raise NameError(
                "A backend file named `%s` could not be found." % backend_file)

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

    return self.bjfunc(chain)


def get_log_prob(self, **args):
    """Get the log likelihoods in the chain
    """
    # just a wrapper
    return get_chain(self, get_log_prob=True, **args)


def write_yaml(self, filename):

    if filename[-5:] != '.yaml':
        filename = filename + '.yaml'

    f = open(filename, "w+")

    f.write(self.raw_yaml)
    f.close()

    print("Model written to '%s.'" % filename)

    return


def save_meta(self, filename=None, verbose=True):

    import os

    filename = filename or os.path.join(self.path, self.name + '_meta')

    objs = 'description', 'backend_file', 'tune', 'name'

    for o in objs:
        if hasattr(self, o):
            exec('self.fdict[o] = self.'+str(o))

    if hasattr(self, 'filter'):
        self.fdict['filter_R'] = self.filter.R
        self.fdict['filter_P'] = self.filter.P

    np.savez(filename, **self.fdict)

    if verbose:
        print('[save_meta:]'.ljust(15, ' ') +
              " Metadata saved as '%s'" % filename)

    return


def save_rdict(self, rdict, path=None, suffix='', verbose=True):
    """Save dictionary of results

    The idea is to keep meta data (the model, setup, ...) and the results obtained (chains, smoothed residuals, ...) separate. 
    """

    if not isinstance(path, str):
        path = self.name + '_res'

    if path[-4] == '.npz':
        path = path[-4]

    if not os.path.isabs(path):
        path = os.path.join(self.path, path)

    np.savez(path + suffix, **rdict)

    if verbose:
        print('[save_rdict:]'.ljust(15, ' ') + " Results saved as '%s'" % path)
    return


def load_rdict(self, path=None, suffix=''):
    """Load stored dictionary of results

    The idea is to keep meta data (the model, setup, ...) and the results obtained (chains, smoothed residuals, ...) separate. `save_rdict` suggests some standard conventions.
    """

    if path is None:
        path = self.name + '_res'

    if path[-4] != '.npz':
        path += '.npz'

    if not os.path.isabs(path):
        path = os.path.join(self.path, path)

    return dict(np.load(path + suffix, allow_pickle=True))


def traceplot_m(self, chain=None, **args):

    if chain is None:
        if 'kdes_chain' in self.fdict.keys():
            chain = self.fdict['kdes_chain']
            args['tune'] = int(chain.shape[0]/5)
        else:
            chain = self.get_chain()
            args['tune'] = get_tune(self)

    return traceplot(chain, varnames=self.fdict['prior_names'], **args)


def posteriorplot_m(self, **args):

    tune = get_tune(self)

    return posteriorplot(self.get_chain(), varnames=self.fdict['prior_names'], tune=tune, **args)


def mode_summary(self, data_cmaes=None, verbose=True):
    """Create a summary of the different results for the mode
    """

    df_inp = {}

    try:
        df_inp['mcmc: mode'] = list(
            self.fdict['mcmc_mode_x']) + [float(self.fdict['mcmc_mode_f'])]
    except KeyError:
        pass

    try:
        data_cmaes = data_cmaes or self.fdict['cmaes_history']
        f_cmaes, x_cmaes = data_cmaes[:2]

        for s, p in enumerate(x_cmaes):
            df_inp['run %s: mode' % s] = list(p) + [f_cmaes[s]]
    except KeyError:
        pass

    df = pd.DataFrame(df_inp)
    df.index = self.prior_names + ['loglike']

    if verbose:
        print(df.round(3))

    return df


def swarm_summary(self, verbose=True, **args):

    res = summary(self, store=self.fdict['swarms'],
                  bounds=self.fdict['prior_bounds'], **args)

    if verbose:
        print(res.round(3))

    return res


def mcmc_summary(self, chain=None, tune=None, calc_mdd=True, calc_ll_stats=False, calc_maf=True, out=print, verbose=True, **args):

    try:
        chain = self.get_chain() if chain is None else chain
    except AttributeError:
        raise AttributeError('[summary:]'.ljust(
            15, ' ') + "No chain to be found...")

    tune = tune or get_tune(self)

    chain = chain[-tune:]
    nchain = chain.reshape(-1, chain.shape[-1])
    lprobs = self.get_log_prob()[-tune:]
    lprobs = lprobs.reshape(-1, lprobs.shape[-1])
    mode_x = nchain[lprobs.argmax()]

    res = summary(self, chain, mode_x, **args)

    if verbose:

        out(res.round(3))

        if calc_mdd:
            out('Marginal data density:' +
                str(mdd(self, tune=tune).round(4)).rjust(16))
        if calc_ll_stats:

            max_lprob = lprobs.max()
            max_lprior = self.lprior(list(mode_x))
            max_llike = (max_lprob - max_lprior) / \
                self.temp if self.temp else np.nan

            out('Max posterior density:' + str(np.round(max_lprob, 4)).rjust(16))
            out('Corresponding likelihood:' +
                str(np.round(max_llike, 4)).rjust(13))
            out('Corresponding prior density:' +
                str(np.round(max_lprior, 4)).rjust(10))
        if calc_maf:

            acs = get_chain(self, get_acceptance_fraction=True)[-tune:]
            out('Mean acceptance fraction:' +
                str(np.mean(acs).round(3)).rjust(13))

    return res


def info_m(self, verbose=True, **args):

    try:
        name = self.name
    except AttributeError:
        name = self.fdict['name']

    try:
        description = self.description
    except AttributeError:
        description = self.fdict['description']

    try:
        dtime = str(self.fdict['datetime'])
    except KeyError:
        dtime = ''

    res = 'Title: %s\n' % name
    res += 'Date: %s\n' % dtime if dtime else ''
    res += 'Description: %s\n' % description

    try:
        cshp = self.get_chain().shape
        tune = get_tune(self)
        res += 'Parameters: %s\n' % cshp[2]
        res += 'Chains: %s\n' % cshp[1]
        res += 'Last %s of %s samples\n' % (tune, cshp[0])
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

    if not isinstance(df, pd.DataFrame):
        raise TypeError('Type of input data must be a `pandas.DataFrame`.')

    if self is not None:
        for o in self['observables']:
            if str(o) not in df.keys():
                raise KeyError('%s is not in the data!' % o)

    if self is not None:
        d = df[self.observables]

    if start is not None:
        start = str(start)

    if end is not None:
        end = str(end)

    d = d.loc[start:end]

    import cloudpickle as cpickle
    self.data = d
    self.fdict['data'] = cpickle.dumps(d)
    self.fdict['obs'] = self.observables

    return d


def lprob(self, par, linear=None, verbose=False):

    if not hasattr(self, 'ndim'):
        self.prep_estim(linear=linear, load_R=True, verbose=verbose)
        linear = self.filter.name == 'KalmanFilter'

    return self.lprob(par, linear=linear, verbose=verbose)


def bjfunc(self, x):

    bnd = np.array(self.fdict['prior_bounds'])

    if not 'biject' in self.fdict.keys():
        return x

    if not self.fdict['biject']:
        return x

    x = 1/(1 + np.exp(x))
    return (bnd[1] - bnd[0])*x + bnd[0]


def rjfunc(self, x):
    bnd = np.array(self.fdict['prior_bounds'])

    if not 'biject' in self.fdict.keys():
        return x

    if not self.fdict['biject']:
        return x

    x = (x - bnd[0])/(bnd[1] - bnd[0])
    return np.log(1/x - 1)


def get_sample(self, size, chain=None):
    """Get a (preferably recent) sample from the chain
    """

    chain = None or self.get_chain()
    clen, nwalks, npar = chain.shape
    recent = int(np.ceil(60 / nwalks))

    if recent > clen:
        raise Exception("Requested sample size is larger than chain")

    sample = chain[:, -recent, :].reshape(-1, npar)
    res = np.random.choice(np.arange(recent*nwalks), size, False)

    return sample[res]


DSGE.vix = vix
DSGE.get_tune = get_tune
DSGE.save = save_meta
DSGE.mapper = mapper
DSGE.mode_summary = mode_summary
DSGE.swarm_summary = swarm_summary
DSGE.mcmc_summary = mcmc_summary
DSGE.info = info_m
DSGE.mdd = mdd
DSGE.get_data = load_data
DSGE.load_data = load_data
DSGE.obs = calc_obs
DSGE.box_check = box_check
DSGE.rjfunc = rjfunc
DSGE.bjfunc = bjfunc
DSGE.get_sample = get_sample
DSGE.create_pool = create_pool
# from core
DSGE.get_par = get_par
DSGE.set_par = set_par
DSGE.get_sys = get_sys
# from tools
DSGE.t_func = t_func
DSGE.o_func = o_func
DSGE.lin_t_func = lin_t_func
DSGE.lin_o_func = lin_o_func
DSGE.get_eps_lin = get_eps_lin
DSGE.irfs = irfs
DSGE.simulate = simulate
DSGE.simulate_ts = simulate_ts
# from mcmc
DSGE.mcmc = mcmc
DSGE.tmcmc = tmcmc
# from estimation
DSGE.prep_estim = prep_estim
DSGE.load_estim = prep_estim
DSGE.lprob = lprob
# from modesearch
DSGE.cmaes = cmaes
# from filter
DSGE.create_filter = create_filter
DSGE.run_filter = run_filter
DSGE.get_ll = get_ll
# from plot
DSGE.traceplot = traceplot_m
DSGE.posteriorplot = posteriorplot_m
# from misc
DSGE.get_chain = get_chain
DSGE.get_log_prob = get_log_prob
DSGE.extract = extract
DSGE.create_obs_cov = create_obs_cov
DSGE.mask = mask
DSGE.load_rdict = load_rdict
DSGE.save_rdict = save_rdict
DSGE.gfevd = gfevd
DSGE.mbcs_index = mbcs_index
DSGE.nhd = nhd
