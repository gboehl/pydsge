#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def write_yaml(self, filename):

    if filename[-5:] is not '.yaml':
        filename = filename + '.yaml'

    f = open(filename, "w+")

    f.write(self.raw_yaml)
    f.close()

    print("Model written to '%s.'" %filename)

    return 


def save_meta(self, filename=None):
    
    if filename is None:
        if 'dfile' in self.fdict.keys():
            filename = self.fdict['dfile']
        else:
            raise KeyError("'filename' must be given.")
    else:
        self.fdict['dfile'] = filename

    if hasattr(self, 'description'):

        self.fdict['description'] = self.description

    if hasattr(self, 'data'):

        self.fdict['data'] = self.data

    np.savez(filename, **self.fdict)

    return


def set_path(self, path):

    import os

    if not path[-1] == os.sep:
        path = path + os.sep
    
    self.path = path

    return


from .parser import DSGE as dsge
from .stuff import *
from .plots import *
from .processing import *
from .estimation import init_estimation, pmdm, estim
from .engine import preprocess, boehlgorithm
from .filtering import *
from .plots import get_iv

dsge.t_func = t_func
dsge.set_path = set_path
dsge.linear_representation = linear_representation
dsge.o_func = o_func
dsge.get_sys = get_sys
dsge.irfs = irfs
dsge.simulate = simulate
dsge.simulate_series = simulate_series
dsge.create_filter = create_filter
dsge.run_filter = run_filter
dsge.get_ll = get_ll
dsge.get_iv = get_iv
dsge.init_estimation = init_estimation
dsge.estim = estim
dsge.pmdm = pmdm
dsge.save = save_res
dsge.epstract = epstract
dsge.sampled_sim = sampled_sim
dsge.sampled_irfs = sampled_irfs
dsge.extract = extract
dsge.create_obs_cov = create_obs_cov
dsge.posterior_sample = posterior_sample
dsge.preprocess = preprocess
dsge.mask = mask
dsge.boehlgorithm = boehlgorithm
dsge.save2 = save_meta

DSGE = dsge


"""
def save_meta(self, filename, save_tune=True, description=None):

    if hasattr(self, 'kf'):
        init_cov = self.kf.P
    else:
        init_cov = self.enkf.P

    if save_tune:
        chain = self.sampler.get_chain()
        tune = self.sampler.tune
    else:
        chain = self.sampler.get_chain()[self.sampler.tune:]
        tune = 0

    np.savez_compressed(filename,
                        Z=self.Z,
                        vv=self.vv,
                        years=self.years,
                        description=self.description,
                        obs_cov=self.obs_cov,
                        init_cov=init_cov,
                        par_fix=self.par_fix,
                        ndraws=self.ndraws,
                        chain=chain,
                        acc_frac=self.sampler.acceptance_fraction,
                        prior_dist=self.sampler.prior_dist,
                        prior_names=self.sampler.prior_names,
                        prior_arg=self.prior_arg,
                        priors=self['__data__']['estimation']['prior'],
                        tune=tune,
                        modelpath=self['filename'],
                        means=self.sampler.par_means)
    print('[save_res:]'.ljust(15, ' ')+'Results saved in ', filename)
    """
