#!/bin/python
# -*- coding: utf-8 -*-

from .parser import DSGE
from .plots import posteriorplot, traceplot, swarm_rank, swarm_champ, swarm_plot
from .mcmc import mcmc, kdes, tmcmc
from .modesearch import pmdm, nlopt, cmaes, cmaes2, swarms
from .filtering import *
from .tools import *
from .core import *
import os
import numpy as np
import pandas as pd
from .stats import summary, pmdm_report


def get_tune(self):

    if hasattr(self, 'tune'):
        return self.tune
    else:
        return self.fdict['tune']


def calc_obs(self, states, covs=None):
    """Get observables from state representation

    Parameters
    ----------
    states : array
    covs : array, optional
        Series of covariance matrices. If provided, 95% intervals will be calculated.
    """

    if covs is None:
        return states @ self.hx[0].T + self.hx[1]

    var = np.diagonal(covs, axis1=1, axis2=2)
    std = np.sqrt(var)
    iv95 = np.stack((states - 1.96*std, states, states + 1.96*std))

    obs = (self.hx[0] @ states.T).T + self.hx[1]
    std_obs = (self.hx[0] @ std.T).T
    iv95_obs = np.stack((obs - 1.96*std_obs, obs, obs + 1.96*std_obs))

    return iv95_obs, iv95


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
        print("'Metadata saved as '%s'" % filename)

    return


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

    res = summary(store=self.fdict['swarms'], priors=self['__data__']
                  ['estimation']['prior'], bounds=self.fdict['prior_bounds'], **args)

    if verbose:
        print(res.round(3))

    return res


def mcmc_summary(self, chain=None, tune=None, calc_mdd=True, calc_ll_stats=False, calc_maf=True, out=print, verbose=True, **args):

    try:
        chain = self.get_chain() if chain is None else chain
    except AttributeError:
        raise AttributeError('[summary:]'.ljust(
            15, ' ') + "No chain to be found...")

    if tune is None:
        tune = get_tune(self)

    res = summary(chain, self['__data__']['estimation']
                  ['prior'], tune=tune, **args)

    if verbose:

        out(res.round(3))

        if calc_mdd:
            out('Marginal data density:' +
                str(mdd(self, tune=tune).round(4)).rjust(16))
        if calc_ll_stats:

            chain = chain[-tune:]
            chain = chain.reshape(-1, chain.shape[-1])
            lprobs = self.sampler.get_log_prob()[-tune:]
            lprobs = lprobs.reshape(-1, lprobs.shape[-1])
            par_lprob = chain[lprobs.argmax()]
            max_lprob = lprobs.max()
            max_lprior = self.lprior(list(par_lprob))
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


def mdd(self, chain=None, mode_f=None, inv_hess=None, tune=None, verbose=False):
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

        if chain is None:
            tune = tune or get_tune(self)
            chain = self.get_chain()[-tune:]
            chain = chain.reshape(-1, chain.shape[-1])

        inv_hess = np.cov(chain.T)

    ndim = len(self.fdict['prior_names'])
    log_det_inv_hess = np.log(np.linalg.det(inv_hess))
    mdd = .5*ndim*np.log(2*np.pi) + .5*log_det_inv_hess + mode_f

    if verbose:
        print('[mdd:]'.ljust(15, ' ') + "Calculation took %s. The marginal data density is %s." %
              (timeprint(time.time()-st), mdd.round(4)))

    return mdd


def box_check(self, par=None):
    """Check if parameterset lies outside the box constraints

    Parameters
    ----------
    par : array or list, optional
        The parameter set to check
    """

    if par is None:
        par = self.par

    for i, name in enumerate(self.fdict['prior_names']):

        lb, ub = self.fdict['prior_bounds']

        if par[i] < lb[i]:
            print('[box_check:]'.ljust(
                15, ' ') + 'Parameter %s of %s lower than lb of %s.' % (name, par[i].round(5), lb[i]))

        if par[i] > ub[i]:
            print('[box_check:]'.ljust(
                15, ' ') + 'Parameter %s of %s higher than ub of %s.' % (name, par[i].round(5), ub[i]))

    return


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


def sample_box(self, dim0, dim1=None, bounds=None, lp_rule=None, verbose=False):
    """Sample from a hypercube
    """
    import chaospy

    bnd = bounds or np.array(self.fdict['prior_bounds'])
    dim1 = dim1 or self.ndim
    rule = lp_rule or 'S'

    res = chaospy.Uniform(0, 1).sample(size=(dim0, dim1), rule=rule)
    res = (bnd[1] - bnd[0])*res + bnd[0]

    return res


@property
def mapper(self):

    if hasattr(self, 'pool'):
        return self.pool.imap
    else:
        return map


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


def create_pool(self, ncores=None):

    import pathos

    self.pool = pathos.pools.ProcessPool(ncores)
    self.pool.clear()

    return self.pool


from .estimation import prep_estim


DSGE.get_tune = get_tune
DSGE.save = save_meta
DSGE.mapper = mapper
DSGE.mode_summary = mode_summary
DSGE.swarm_summary = swarm_summary
DSGE.mcmc_summary = mcmc_summary
DSGE.info = info_m
DSGE.pmdm_report = pmdm_report
DSGE.mdd = mdd
DSGE.get_data = load_data
DSGE.load_data = load_data
DSGE.obs = calc_obs
DSGE.box_check = box_check
DSGE.rjfunc = rjfunc
DSGE.bjfunc = bjfunc
DSGE.sample_box = sample_box
DSGE.get_sample = get_sample
DSGE.create_pool = create_pool
# from core & tools:
DSGE.get_par = get_par
DSGE.set_par = set_par
DSGE.get_sys = get_sys
DSGE.t_func = t_func
DSGE.o_func = o_func
DSGE.get_eps = get_eps
DSGE.irfs = irfs
DSGE.simulate = simulate
DSGE.linear_representation = linear_representation
DSGE.simulate_ts = simulate_ts
# from estimation:
DSGE.swarms = swarms
DSGE.cmaes = cmaes
DSGE.cmaes2 = cmaes2
DSGE.mcmc = mcmc
DSGE.tmcmc = tmcmc
DSGE.kdes = kdes
DSGE.kombine = kdes
DSGE.prep_estim = prep_estim
DSGE.load_estim = prep_estim
DSGE.lprob = lprob
# from modesearch:
DSGE.pmdm = pmdm
DSGE.nlopt = nlopt
# from filter:
DSGE.create_filter = create_filter
DSGE.run_filter = run_filter
# from plot:
DSGE.traceplot = traceplot_m
DSGE.posteriorplot = posteriorplot_m
DSGE.swarm_champ = swarm_champ
DSGE.swarm_plot = swarm_plot
DSGE.swarm_rank = swarm_rank
# from others:
DSGE.get_chain = get_chain
DSGE.get_log_prob = get_log_prob
DSGE.extract = extract
DSGE.create_obs_cov = create_obs_cov
DSGE.mask = mask
DSGE.load_eps = load_eps
