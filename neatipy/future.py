#!/bin/python2
# -*- coding: utf-8 -*-

## Bayesian Estimation using pymc3. Not operational for now (and probably never)
def be_pymc(self, alpha = 0.2, scale_obs = 0.2, draws = 500, tune = 500, ncores = None, use_find_MAP = True, info = False):

    import pymc3 as pm
    import theano.tensor as tt
    from theano.compile.ops import as_op
    import multiprocessing

    if ncores is None:
        # ncores    = multiprocessing.cpu_count() - 1
        ncores    = 4

    ## dry run before the fun beginns
    self.create_filter(scale_obs = scale_obs)
    self.ukf.R[-1,-1]  /= 100
    self.run_filter()
    print("Model operational. Ready for estimation.")

    ## from here different from filtering
    par_active  = np.array(self.par).copy()

    p_names     = [ p.name for p in self.parameters ]
    priors      = self['__data__']['estimation']['prior']
    prior_arg   = [ p_names.index(pp) for pp in priors.keys() ]

    init_par    = dict(zip(np.array(p_names)[prior_arg], np.array(self.par)[prior_arg]))

    tlist   = []
    for i in range(len(priors)):
        tlist.append(tt.dscalar)

    @as_op(itypes=tlist, otypes=[tt.dvector])
    def get_ll(*parameters):
        st  = time.time()

        try: 
            par_active[prior_arg]  = parameters
            par_active_lst  = list(par_active)

            self.get_sys(par_active_lst)
            self.preprocess(info=info)

            self.create_filter(scale_obs = scale_obs)
            self.ukf.R[-1,-1]  /= 100
            self.run_filter(info=info)

            if info == 2:
                print('Sample took '+str(np.round(time.time() - st))+'s.')
            return self.ll.reshape(1)

        except:

            if info == 2:
                print('Sample took '+str(np.round(time.time() - st))+'s. (failure)')
            return np.array(-sys.maxsize - 1, dtype=float).reshape(1)


    with pm.Model() as model:
        
        be_pars_lst     = []
        for pp in priors:
            dist    = priors[str(pp)]
            pmean = dist[1]
            pstdd = dist[2]

            if str(dist[0]) == 'uniform':
                be_pars_lst.append( pm.Uniform(str(pp), lower=dist[1], upper=dist[2]) )
            elif str(dist[0]) == 'inv_gamma':
                alp     = pmean**2/pstdd**2 + 2
                bet     = pmean*(alp - 1)
                be_pars_lst.append( pm.InverseGamma(str(pp), alpha=alp, beta=bet) )
            elif str(dist[0]) == 'normal':
                be_pars_lst.append( pm.Normal(str(pp), mu=pmean, sd=pstdd) )
            elif str(dist[0]) == 'gamma':
                be_pars_lst.append( pm.Gamma(str(pp), mu=pmean, sd=pstdd) )
            elif str(dist[0]) == 'beta':
                be_pars_lst.append( pm.Beta(str(pp), mu=pmean, sd=pstdd) )
            else:
                print('Distribution not implemented')
            print('Adding parameter %s as %s to the prior distributions.' %(pp, dist[0]))

        be_pars = tuple(be_pars_lst)

        pm.Potential('logllh', get_ll(*be_pars))
        
        if use_find_MAP:
            self.MAP = pm.find_MAP(start=init_par)
        else:
            self.MAP = init_par
        step = pm.Metropolis()
        self.trace = pm.sample(draws=draws, tune=tune, step=step, start=self.MAP, cores=ncores, random_seed=list(np.arange(ncores)))

    return be_pars
