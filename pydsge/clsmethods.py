#!/bin/python
# -*- coding: utf-8 -*-

import emcee
import numpy as np
import pandas as pd
from .parser import DSGE
from .stats import summary, pmdm_report
from .engine import preprocess, func_dispatch
from .stuff import *
from .filtering import *
from .estimation import prep_estim, swarms, mcmc, kdes, get_par
from .modesearch import pmdm, nlopt
from .plots import posteriorplot, traceplot, swarm_rank, swarm_champ, swarm_plot
from .processing import *


@property
def get_tune(self):

    if hasattr(self, 'tune'):
        return self.tune
    else:
        return self.fdict['tune']


def calc_obs(self, states, covs=None):

    if covs is None:
        return states @ self.hx[0].T + self.hx[1]

    var = np.diagonal(covs, axis1=1, axis2=2)
    std = np.sqrt(var)
    iv95 = np.stack((states - 1.96*std, states, states + 1.96*std))

    obs = (self.hx[0] @ states.T).T + self.hx[1]
    std_obs = (self.hx[0] @ std.T).T
    iv95_obs = np.stack((obs - 1.96*std_obs, obs, obs + 1.96*std_obs))

    return iv95_obs, iv95


def get_chain(self, get_acceptance_fraction=False, mc_type=None, backend_file=None, flat=None):

    if backend_file is None:

        if hasattr(self, 'kdes_sample') and mc_type != 'mcmc':
            return self.kdes_sample

        if hasattr(self, 'sampler'):
            return self.sampler.get_chain(flat=flat)

        if 'kdes_sample' in self.fdict.keys() and mc_type != 'mcmc':
            return self.fdict['kdes_sample']

        if hasattr(self, 'backend_file'):
            backend_file = self.backend_file
        elif 'backend_file' in self.fdict.keys():
            backend_file = str(self.fdict['backend_file'])
        else:
            raise AttributeError(
                "Neither a backend nor a sampler could be found.")

    reader = emcee.backends.HDFBackend(backend_file)

    if get_acceptance_fraction:
        return reader.accepted / reader.iteration

    return reader.get_chain(flat=flat)


def get_log_prob(self, mc_type=None, backend_file=None, flat=None):

    if backend_file is None:

        if hasattr(self, 'sampler'):
            return self.sampler.get_log_prob(flat=flat)

        if 'backend_file' in self.fdict.keys():
            backend_file = str(self.fdict['backend_file'])
        else:
            raise AttributeError(
                "Neither a backend nor a sampler could be found.")

    reader = emcee.backends.HDFBackend(backend_file)

    return reader.get_log_prob(flat=flat)


def write_yaml(self, filename):

    if filename[-5:] is not '.yaml':
        filename = filename + '.yaml'

    f = open(filename, "w+")

    f.write(self.raw_yaml)
    f.close()

    print("Model written to '%s.'" % filename)

    return


def save_meta(self, filename=None, verbose=True):

    if filename is None:
        if hasattr(self, 'path') and hasattr(self, 'name'):
            filename = self.path + self.name + '_meta'
        elif hasattr(self, 'path') and hasattr(self, 'mod_name'):
            filename = self.path + self.mod_name + '_meta'
        elif 'dfile' in self.fdict.keys():
            filename = self.fdict['dfile']
        else:
            raise KeyError("'filename' must be given.")
    else:
        self.fdict['dfile'] = filename

    objs = 'description', 'backend_file', 'tune', 'name'

    for o in objs:
        if hasattr(self, o):
            exec('self.fdict[o] = self.'+str(o))

    if hasattr(self, 'filter'):
        self.fdict['filter_R'] = self.filter.R
        self.fdict['filter_P'] = self.filter.P

    np.savez(filename, **self.fdict)

    if verbose:
        print("'Metadata saved as '%s'" % filename)

    return


def set_path(self, path):

    import os

    if not path[-1] == os.sep:
        path = path + os.sep

    self.path = path

    return


def traceplot_m(self, chain=None, **args):

    if chain is None:
        if 'kdes_chain' in self.fdict.keys():
            chain = self.fdict['kdes_chain']
            args['tune'] = int(chain.shape[0]*4/5)
        else:
            chain = self.get_chain()
            args['tune'] = self.get_tune

    return traceplot(chain, varnames=self.fdict['prior_names'], **args)


def posteriorplot_m(self, mc_type=None, **args):

    tune = self.get_tune

    return posteriorplot(self.get_chain(mc_type=mc_type), varnames=self.fdict['prior_names'], tune=tune, **args)


def swarm_summary(self, verbose=True, **args):

    res = summary(store=self.fdict['swarms'], priors=self['__data__']
                  ['estimation']['prior'], bounds=self.fdict['prior_bounds'], **args)

    if verbose:
        print(res.round(3))

    return res


def mcmc_summary(self, mc_type=None, tune=None, verbose=True, **args):

    try:
        chain = self.get_chain(mc_type)
    except AttributeError:
        raise AttributeError('[summary:]'.ljust(
            15, ' ') + "No chain to be found...")

    res = summary(chain, self['__data__']['estimation']
                  ['prior'], tune=tune, **args)

    if tune is None:
        tune = self.get_tune

    if verbose:
        acs = get_chain(self, get_acceptance_fraction=True)
        print(res.round(3))
        print('Marginal data density:' + str(mdd(self).round(4)).rjust(16))
        print('Mean acceptance fraction:' + str(np.mean(acs).round(3)).rjust(13))

            # raise ValueError('[mdd:]'.ljust(
                # 15, ' ') + "Option `hess` is experimental and did not return a usable hessian matrix.")
    return res


def info_m(self, verbose=True, **args):

    if hasattr(self, 'name'):
        name = self.name
    else:
        name = self.fdict['name']

    if hasattr(self, 'description'):
        description = self.description
    else:
        description = self.fdict['description']

    res = 'Title: %s\n' %name
    res += 'Description: %s\n' %description

    try:
        cshp = self.get_chain().shape
        tune = self.get_tune
        res += 'Parameters: %s\n' %cshp[2]
        res += 'Chains: %s\n' %cshp[1]
        res += 'Last %s of %s samples\n' %(cshp[0] - tune, cshp[0])
    except AttributeError:
        pass

    if verbose:
        print(res)

    return res


def get_data(self=None, csv=None, sep=None, start=None, end=None):

    if csv[-4:] != '.csv' or csv is None:
        raise TypeError('data format must be `.csv`.')

    d = pd.read_csv(csv, sep=sep).dropna()

    if self is not None:
        for o in self['observables']:
            if str(o) not in d.keys():
                raise KeyError('%s is not in the data!' % o)

    dates = pd.date_range(
        str(int(d['year'][0])), periods=d.shape[0], freq='Q')

    d.index = dates

    if self is not None:
        self.obs = [str(o) for o in self['observables']]
        d = d[self.obs]

    if start is not None:
        start = str(start)

    if end is not None:
        end = str(end)

    d = d.loc[start:end]

    if self is not None:
        import dill
        self.data = d
        self.fdict['data'] = dill.dumps(d)
        self.fdict['obs'] = self.obs

    return d


def lprob(self, par, linear=None, verbose=False):

    if not hasattr(self, 'ndim'):
        self.prep_estim(linear=linear, load_R=True, verbose=verbose)
        linear = self.filter.name == 'KalmanFilter'

    return self.lprob(par, linear=linear, verbose=verbose)


def mdd(self, mode_f=None, inv_hess=None, verbose=False):
    """Approximate the marginal data density useing the LaPlace method.
    `inv_hess` can be a matrix or the method string in ('hess', 'cov') telling me how to Approximate the inverse Hessian
    """

    if verbose:
        st = time.time()

    if mode_f is None:
        mode_f = self.fdict['mode_f']

    if inv_hess == 'hess':

        import numdifftools as nd

        np.warnings.filterwarnings('ignore')
        hh = nd.Hessian(func)(self.fdict['mode_x'])
        np.warnings.filterwarnings('default')

        if np.isnan(hh).any():
            raise ValueError('[mdd:]'.ljust(
                15, ' ') + "Option `hess` is experimental and did not return a usable hessian matrix.")

        inv_hess = np.linalg.inv(hh)

    elif inv_hess is None:

        chain = self.get_chain()[self.get_tune:]
        chain = chain.reshape(-1, chain.shape[-1])
        inv_hess = np.cov(chain.T)

    ndim = len(self.fdict['prior_names'])
    log_det_inv_hess = np.log(np.linalg.det(inv_hess))
    mdd = .5*ndim*np.log(2*np.pi) + .5*log_det_inv_hess + mode_f

    if verbose:
        print('[mdd:]'.ljust(15, ' ') + "Calculation took %s. The marginal data density is %s." %
              (timeprint(time.time()-st), mdd.round(4)))

    return mdd


DSGE.save = save_meta
DSGE.swarm_summary = swarm_summary
DSGE.mcmc_summary = mcmc_summary
DSGE.info = info_m
DSGE.pmdm_report = pmdm_report
DSGE.mdd = mdd
DSGE.get_data = get_data
DSGE.get_tune = get_tune
DSGE.get_par = get_par
DSGE.calc_obs = calc_obs
DSGE.obs = calc_obs
# from stuff:
DSGE.func_dispatch = func_dispatch
DSGE.get_sys = get_sys
DSGE.get_calib = get_calib
DSGE.set_calib = set_calib
DSGE.t_func = t_func
DSGE.o_func = o_func
DSGE.get_eps = get_eps
DSGE.irfs = irfs
DSGE.simulate = simulate
DSGE.linear_representation = linear_representation
DSGE.simulate_series = simulate_series
# from estimation:
DSGE.swarms = swarms
DSGE.mcmc = mcmc
DSGE.kdes = kdes
DSGE.kombine = kdes
DSGE.prep_estim = prep_estim
DSGE.lprob = lprob
# from modesearch:
DSGE.pmdm = pmdm
DSGE.nlopt = nlopt
# from filter:
DSGE.create_filter = create_filter
DSGE.run_filter = run_filter
DSGE.get_ll = get_ll
# from plot:
DSGE.traceplot = traceplot_m
DSGE.posteriorplot = posteriorplot_m
DSGE.swarm_champ = swarm_champ
DSGE.swarm_plot = swarm_plot
DSGE.swarm_rank = swarm_rank
# from others:
DSGE.set_path = set_path
DSGE.get_chain = get_chain
DSGE.get_log_prob = get_log_prob
DSGE.extract = extract
DSGE.create_obs_cov = create_obs_cov
DSGE.preprocess = preprocess
DSGE.mask = mask
DSGE.sampled_extract = sampled_extract
DSGE.sampled_sim = sampled_sim
DSGE.sampled_irfs = sampled_irfs

"""
def chain_masker(self):
    iss = np.zeros(len(self.prior_names), dtype=bool)
    for v in self.prior_names:
        iss = iss | (self.prior_names == v)
    return iss
"""
