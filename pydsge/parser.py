#!/bin/python
# -*- coding: utf-8 -*-

import os
import re
import yaml
import itertools
import sympy
import time
import numpy as np
import scipy.stats as sst
import scipy.optimize as sso
import cloudpickle as cpickle
from sys import platform
from copy import deepcopy
from .symbols import Variable, Equation, Shock, Parameter, TSymbol
from sympy.matrices import Matrix, zeros


class DSGE(dict):
    """Base class. Every model is an instance of the DSGE class and inherents its methods.
    """

    def __init__(self, *kargs, **kwargs):
        super(DSGE, self).__init__(self, *kargs, **kwargs)

        fvars = []
        lvars = []

        # get forward looking variables
        for eq in self['equations']:

            variable_too_far = [v for v in eq.atoms(
            ) if isinstance(v, Variable) and v.date > 1]
            variable_too_early = [v for v in eq.atoms(
            ) if isinstance(v, Variable) and v.date < -1]

            eq_fvars = [v for v in eq.atoms() if isinstance(
                v, TSymbol) and v.date > 0]
            eq_lvars = [v for v in eq.atoms() if isinstance(
                v, TSymbol) and v.date < 0]

            fvars = list(set(fvars).union(eq_fvars))
            lvars = list(set(lvars).union(set(eq_lvars)))

        self['info']['nstate'] = len(self.variables) + len(fvars)

        self['fvars'] = fvars
        self['fvars_lagged'] = [Parameter('__LAGGED_'+f.name) for f in fvars]
        self['lvars'] = lvars
        self['re_errors'] = [Shock('eta_'+v.name) for v in self['fvars']]

        self['re_errors_eq'] = []
        i = 0
        for fv, lag_fv in zip(fvars, self['fvars_lagged']):
            self['re_errors_eq'].append(
                Equation(fv(-1) - lag_fv - self['re_errors'][i], 0))
            i += 1

        # if 'make_log' in self.keys():
            # self['perturb_eq'] = []
            # sub_dict = dict()
            # sub_dict.update({v: Variable(v.name+'ss')*sympy.exp(v)
            # for v in self['make_log']})
            # sub_dict.update({v(-1): Variable(v.name+'ss') *
            # sympy.exp(v(-1)) for v in self['make_log']})
            # sub_dict.update({v(1): Variable(v.name+'ss')*sympy.exp(v(1))
            # for v in self['make_log']})

            # for eq in self.equations:
            # peq = eq.subs(sub_dict)
            # self['perturb_eq'].append(peq)

            # self['ss_ordering'] = [Variable(v.name+'ss')
            # for v in self['make_log']]

        # else:
            # self['perturb_eq'] = self['equations']
        self['perturb_eq'] = self['equations']

        context = {}

        return

    def __repr__(self):
        return "A DSGE Model."

    @property
    def equations(self):
        return self['equations']

    @property
    def variables(self):
        return self['var_ordering']

    # ->
    @property
    def const_var(self):
        return self['const_var']

    @property
    def const_eq(self):
        return self['const_eq']
    # <-

    @property
    def parameters(self):
        return self['par_ordering']

    @property
    def par_names(self):
        return [p.name for p in self['par_ordering']]

    @property
    def shocks(self):
        return [str(s) for s in self['shk_ordering']]

    @property
    def mod_name(self):
        return self['mod_name']

    @property
    def neq(self):
        return len(self['perturb_eq'])

    @property
    def neq_fort(self):
        return self.neq+self.neta

    @property
    def neta(self):
        return len(self['fvars'])

    @property
    def ny(self):
        return len(self['observables'])

    @property
    def neps(self):
        return len(self['shk_ordering'])

    @property
    def npara(self):
        return len(self.parameters)

    def p0(self):
        return list(map(lambda x: self['calibration'][str(x)], self.parameters))

    def get_matrices(self, matrix_format='numeric'):

        from sympy.utilities.lambdify import lambdify, implemented_function

        # check for uniqueness
        nvars = []
        for v in self['var_ordering']:
            if v in nvars:
                print('[DSGE.read:]'.ljust(15, ' ') +
                      'Warning: variable `%s` defined twice.' % v)
            else:
                nvars.append(v)
        self['var_ordering'] = nvars

        vlist = self['var_ordering'] + self['fvars']
        llist = [l(-1) for l in self['var_ordering']] + self['fvars_lagged']

        slist = self['shk_ordering']

        subs_dict = {}

        eq_cond = self['perturb_eq'] + self['re_errors_eq']

        sub_var = self['var_ordering']
        subs_dict.update({v: 0 for v in sub_var})
        subs_dict.update({v(1): 0 for v in sub_var})
        subs_dict.update({v(-1): 0 for v in sub_var})

        svar = len(vlist)
        evar = len(slist)
        rvar = len(self['re_errors'])
        ovar = len(self['observables'])

        # PSI  = zeros(svar, evar)
        # PPI  = zeros(svar, rvar)

        # ->
        sub_var = self['var_ordering']
        fvarl = [l(+1) for l in sub_var]
        lvarl = [l(-1) for l in sub_var]

        no_var = len(sub_var)
        no_lvar = len(lvarl)

        bb = zeros(1, no_var+no_lvar)

        if self['const_var']:
            AA = zeros(no_var-1, no_var)
            BB = zeros(no_var-1, no_var)
            CC = zeros(no_var-1, no_var)
            PSI = zeros(no_var-1, evar)
            bb_var = filter(lambda x: x.date <= 0,
                            self['const_eq'].atoms(Variable))

            full_var = sub_var + lvarl

            for v in bb_var:
                v_j = full_var.index(v)
                bb[v_j] = -(self['const_eq']
                            ).set_eq_zero.diff(v).subs(subs_dict)
        else:
            AA = zeros(no_var, no_var)
            BB = zeros(no_var, no_var)
            CC = zeros(no_var, no_var)
            PSI = zeros(no_var, evar)

        eq_i = 0
        for eq in self['perturb_eq']:
            # ->
            A_var = filter(lambda x: x.date > 0, eq.atoms(Variable))

            for v in A_var:
                v_j = fvarl.index(v)
                AA[eq_i, v_j] = -(eq).set_eq_zero.diff(v).subs(subs_dict)

            B_var = filter(lambda x: x.date == 0, eq.atoms(Variable))

            for v in B_var:
                v_j = sub_var.index(v)
                BB[eq_i, v_j] = (eq).set_eq_zero.diff(v).subs(subs_dict)

            C_var = filter(lambda x: x.date < 0, eq.atoms(Variable))

            for v in C_var:
                v_j = lvarl.index(v)
                CC[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict)

            shocks = filter(lambda x: x, eq.atoms(Shock))

            for s in shocks:
                s_j = slist.index(s)
                PSI[eq_i, s_j] = -eq.set_eq_zero.diff(s).subs(subs_dict)

            eq_i += 1
            # <-

        DD = zeros(ovar, 1)
        ZZ = zeros(ovar, no_var)

        eq_i = 0
        for obs in self['observables']:
            eq = self['obs_equations'][str(obs)]
            DD[eq_i, 0] = eq.subs(subs_dict)

            curr_var = filter(lambda x: x.date >= 0, eq.atoms(Variable))

            for v in curr_var:
                v_j = vlist.index(v)
                ZZ[eq_i, v_j] = eq.diff(v).subs(subs_dict)

                if self.const_var is v:
                    self.const_obs = obs

            eq_i += 1

        # print ""
        from collections import OrderedDict
        # subs_dict = []
        context = dict([(p.name, p) for p in self.parameters])
        # context['exp'] = sympy.exp
        # context['log'] = sympy.log
        # context['sqrt'] = sympy.sqrt

        sol_dict = {}

        def ctf_reducer(f):
            """hack to reduce the calls-to-function
            """

            def reducer(*x):
                try:
                    return sol_dict[f, x]
                except KeyError:
                    res = f(*x)
                    sol_dict[f, x] = res
                    return res

            return reducer

        # standard functions
        context['exp'] = implemented_function('exp', np.exp)
        context['log'] = implemented_function('log', np.log)
        context['sqrt'] = implemented_function('sqrt', np.sqrt)

        # distributions
        context['normpdf'] = implemented_function(
            'normpdf', ctf_reducer(sst.norm.pdf))
        context['normcdf'] = implemented_function(
            'normcdf', ctf_reducer(sst.norm.cdf))
        context['normppf'] = implemented_function(
            'normppf', ctf_reducer(sst.norm.ppf))
        context['norminv'] = implemented_function(
            'norminv', ctf_reducer(sst.norm.ppf))

        # things defined in *_funcs.py
        if self.func_file and os.path.exists(self.func_file):
            import importlib.util as iu
            import inspect

            spec = iu.spec_from_file_location("module", self.func_file)
            module = iu.module_from_spec(spec)
            spec.loader.exec_module(module)

            funcs_list = [o for o in inspect.getmembers(
                module) if inspect.isfunction(o[1])]

            for func in funcs_list:
                context[func[0]] = implemented_function(
                    func[0], ctf_reducer(func[1]))

        # context_f = {}
        # context_f['exp'] = np.exp
        # if 'helper_func' in self['__data__']['declarations']:
            # from imp import load_source
            # f = self['__data__']['declarations']['helper_func']['file']
            # module = load_source('helper_func', f)
            # for n in self['__data__']['declarations']['helper_func']['names']:
                # context[n] = sympy.Function(n)  # getattr(module, n)
                # context_f[n] = getattr(module, n)
        ss = {}

        # print(context['lamp_p'])
        checker = np.zeros_like(self['other_para'], dtype=bool)
        suc_loop = True

        while ~checker.all():

            # raise if loop was unsuccessful
            raise_error = not suc_loop
            suc_loop = False  # set to check if any progress in loop
            for i, p in enumerate(self['other_para']):
                if not checker[i]:
                    try:
                        ss[str(p)] = eval(
                            str(self['para_func'][p.name]), context)
                        context[str(p)] = ss[str(p)]
                        checker[i] = True
                        suc_loop = True  # loop was successful
                    except NameError as e:
                        if raise_error:
                            error_msg = str(e)
                            if not os.path.exists(self.func_file):
                                fname = os.path.basename(self.func_file)
                                error_msg += ' (info: a file named `%s` was not found)' % fname
                                raise NameError(
                                    "Definitions of `para_func` seem to be circular. Last error: "+error_msg)

        # print(context)
        # DD = DD.subs(subs_dict)
        # ZZ = ZZ.subs(subs_dict)

        DD = lambdify([self.parameters+self['other_para']], DD)
        ZZ = lambdify([self.parameters+self['other_para']], ZZ)

        # , modules={'ImmutableDenseMatrix': np.array})#'numpy')
        PSI = lambdify([self.parameters+self['other_para']], PSI)
        # PPI = lambdify([self.parameters+self['other_para']], PPI)#, modules={'ImmutableDenseMatrix': np.array})#'numpy')

        # ->
        # , modules={'ImmutableDenseMatrix': np.array})#'numpy')
        AA = lambdify([self.parameters+self['other_para']], AA)
        # , modules={'ImmutableDenseMatrix': np.array})#'numpy')
        BB = lambdify([self.parameters+self['other_para']], BB)
        # , modules={'ImmutableDenseMatrix': np.array})#'numpy')
        CC = lambdify([self.parameters+self['other_para']], CC)
        # , modules={'ImmutableDenseMatrix': np.array})#'numpy')
        bb = lambdify([self.parameters+self['other_para']], bb)
        # <-

        psi = lambdify([self.parameters], [ss[str(px)]
                                           for px in self['other_para']])  # , modules=context_f)

        # disable this
        def add_para_func(f):
            return f

        def full_compile(px):
            return list(px) + psi(list(px))

        self.compile = full_compile
        self.parafunc = [p.name for p in self['other_para']], psi
        self.psi = psi
        self.PSI = add_para_func(PSI)

        self.DD = add_para_func(DD)
        self.ZZ = add_para_func(ZZ)
        # ->
        self.AA = add_para_func(AA)
        self.BB = add_para_func(BB)
        self.CC = add_para_func(CC)
        self.bb = add_para_func(bb)
        # <-

        # QQ = self['covariance'].subs(subs_dict)
        # HH = self['measurement_errors'].subs(subs_dict)

        QQ = lambdify([self.parameters+self['other_para']], self['covariance'])
        HH = lambdify([self.parameters+self['other_para']],
                      self['measurement_errors'])

        self.QQ = add_para_func(QQ)
        self.HH = add_para_func(HH)

    @classmethod
    def read(cls, mfile, verbose=False):
        """Read and parse a given `*.yaml` file.

        Parameters
        ----------
        mfile : str
            Path to the `*.yaml` file.
        """

        global processed_raw_model

        if verbose:
            st = time.time()

        f = open(mfile)
        mtxt = f.read()
        f.close()

        use_cached = False

        if 'processed_raw_model' in globals():
            use_cached = processed_raw_model.fdict['yaml_raw'] == mtxt

        if use_cached:
            pmodel = deepcopy(processed_raw_model)

        else:

            func_file = mfile[:-5] + '_funcs.py'

            pmodel = cls.parse(mtxt, func_file)

            pmodel.fdict = {}
            pmodel.fdict['yaml_raw'] = mtxt

            if os.path.exists(func_file):
                ff = open(func_file)
                ftxt = ff.read()
                pmodel.fdict['ffile_raw'] = ftxt

            pmodel_dump = cpickle.dumps(pmodel, protocol=4)
            pmodel.fdict['model_dump'] = pmodel_dump
            pmodel.name = pmodel.mod_name
            pmodel.path = os.path.dirname(mfile)
            pmodel.debug = platform == "darwin" or platform == "win32"
            if pmodel.debug:
                print('[DSGE:]'.ljust(
                    15, ' ') + 'Parallelization disabled under Windows and Mac due to a problem with pickling some of the symbolic elements. Sorry...')

            processed_raw_model = deepcopy(pmodel)

        if verbose:
            duration = np.round(time.time()-st, 3)
            if duration < .01:
                duration = 'the speed of light'
            else:
                str(duration) + 's'
            print('[DSGE:]'.ljust(15, ' ') +
                  'Reading and parsing done in %s.' % duration)

        return pmodel

    @classmethod
    def load(cls, npzfile, force_parse=False, verbose=False):

        global processed_raw_model

        if verbose:
            st = time.time()

        fdict = dict(np.load(npzfile, allow_pickle=True))

        mtxt = str(fdict['yaml_raw'])

        try:
            if force_parse:
                raise Exception
            pmodel = cpickle.loads(fdict['model_dump'])
        except:
            use_cached = False

            if 'processed_raw_model' in globals():
                use_cached = processed_raw_model.fdict['yaml_raw'] == mtxt

            if use_cached:
                pmodel = deepcopy(processed_raw_model)
            else:
                import tempfile

                try:
                    # cumbersome: load the text of the *_funcs file, write it to a temporary file, just to use it as a module
                    ftxt = str(fdict['ffile_raw'])
                    tfile = tempfile.NamedTemporaryFile(
                        'w', suffix='.py', delete=False)
                    tfile.write(ftxt)
                    tfile.close()
                    ffile = tfile.name
                except KeyError:
                    ffile = ''

                pmodel = cls.parse(mtxt, ffile)

                try:
                    tfile.close()
                    os.unlink(tfile.name)
                except:
                    pass

        pmodel.fdict = fdict
        pmodel.name = str(fdict['name'])
        pmodel.path = os.path.dirname(npzfile)
        pmodel.data = cpickle.loads(fdict['data'])
        pmodel.debug = platform == "darwin" or platform == "win32"
        if pmodel.debug:
            print('[DSGE:]'.ljust(
                15, ' ') + 'Parallelization disabled under Windows and Mac due to a problem with pickling some of the symbolic elements. Sorry...')

        for ob in pmodel.fdict.keys():
            if str(pmodel.fdict[ob]) == 'None':
                pmodel.fdict[ob] = None

        if verbose:
            print('[DSGE:]'.ljust(15, ' ')+'Loading and parsing done in %ss.' %
                  np.round(time.time()-st, 5))

        return pmodel

    @classmethod
    def parse(cls, mtxt, ffile):
        """

        """

        mtxt = mtxt.replace('^', '**')
        mtxt = mtxt.replace(';', '')
        mtxt = re.sub(r"@ ?\n", " ", mtxt)
        model_yaml = yaml.safe_load(mtxt)

        dec = model_yaml['declarations']
        cal = model_yaml['calibration']

        name = dec['name']

        var_ordering = [Variable(v) for v in dec['variables']]
        par_ordering = [Parameter(v) for v in dec['parameters']]
        shk_ordering = [Shock(v) for v in dec['shocks']]

        if 'para_func' in dec:
            other_para = [Parameter(v) for v in dec['para_func']]
        else:
            other_para = []

        context = [(s.name, s) for s in var_ordering +
                   par_ordering + shk_ordering + other_para]
        context = dict(context)

        if 'observables' in dec:
            observables = [Variable(v) for v in dec['observables']]
        else:
            observables = []
        if 'observables' in model_yaml['equations']:
            obs_equations = model_yaml['equations']['observables']
        else:
            obs_equations = dict()

        if 'constrained' in dec:
            # c_var       = [Variable(v) for v in dec['constrained']]
            c_var = Variable(dec['constrained'][0])
            # only one constraint allowed
            raw_const = model_yaml['equations']['constraint'][0]

            if '=' in raw_const:
                lhs, rhs = str.split(raw_const, '=')
            else:
                lhs, rhs = raw_const, '0'

            try:
                lhs = eval(lhs, context)
                rhs = eval(rhs, context)
            except TypeError as e:
                raise SyntaxError(
                    'While parsing %s, got this error: %s' % (raw_const, repr(e)))

            const_eq = Equation(lhs, rhs)
        else:
            c_var = []
            const_eq = dict()

        if 'measurement_errors' in dec:
            measurement_errors = [Shock(v) for v in dec['measurement_errors']]
        else:
            measurement_errors = None

        if 'make_log' in dec:
            make_log = [Variable(v) for v in dec['make_log']]
        else:
            make_log = []

        steady_state = [0]
        init_values = [0]

        # for f in [sympy.log, sympy.exp,
        # sympy.sin, sympy.cos, sympy.tan,
        # sympy.asin, sympy.acos, sympy.atan,
        # sympy.sinh, sympy.cosh, sympy.tanh,
        # sympy.pi, sympy.sign]:
        # context[str(f)] = f

        # context['sqrt'] = sympy.sqrt
        context['__builtins__'] = None

        equations = []

        if 'model' in model_yaml['equations']:
            raw_equations = model_yaml['equations']['model']
        else:
            raw_equations = model_yaml['equations']

        for eq in raw_equations:
            if '=' in eq:
                lhs, rhs = str.split(eq, '=')
            else:
                lhs, rhs = eq, '0'

            try:
                lhs = eval(lhs, context)
                rhs = eval(rhs, context)
            except TypeError as e:
                raise SyntaxError(
                    'While parsing %s, got this error: %s' % (eq, repr(e)))

            equations.append(Equation(lhs, rhs))

        # ------------------------------------------------------------
        # Figure out max leads and lags
        # ------------------------------------------------------------
        it = itertools.chain.from_iterable

        # all_shocks_pre = [list(eq.atoms(Shock)) for eq in equations]

        # for s in shk_ordering:
        # max_lag = min(
        # [i.date for i in it(all_shocks_pre) if i.name == s.name])
        # for t in np.arange(max_lag, 1):
        # subs_dict = {s(t): s(t-1)}
        # equations = [eq.subs(subs_dict) for eq in equations]

        max_lead_exo = dict.fromkeys(shk_ordering)
        max_lag_exo = dict.fromkeys(shk_ordering)

        all_shocks = [list(eq.atoms(Shock)) for eq in equations]

        for s in shk_ordering:
            try:
                max_lead_exo[s] = max(
                    [i.date for i in it(all_shocks) if i.name == s.name])
                max_lag_exo[s] = min(
                    [i.date for i in it(all_shocks) if i.name == s.name])
            except Exception as ex:
                raise SyntaxError("While parsing shock '%s': %s" % (s, ex))

        # arbitrary lags of exogenous shocks
        for s in shk_ordering:
            if abs(max_lag_exo[s]) > 0:
                var_s = Variable(s.name+"_VAR")
                var_ordering.append(var_s)
                equations.append(Equation(var_s, s))

                subs1 = [s(-i) for i in np.arange(1, abs(max_lag_exo[s])+1)]
                subs2 = [var_s(-i-1)
                         for i in np.arange(1, abs(max_lag_exo[s])+1)]
                subs_dict = dict(zip(subs1, subs2))
                equations = [eq.subs(subs_dict) for eq in equations]

        all_vars = [list(eq.atoms(Variable)) for eq in equations]
        max_lead_endo = dict.fromkeys(var_ordering)
        max_lag_endo = dict.fromkeys(var_ordering)

        for v in var_ordering:
            try:
                max_lead_endo[v] = max(
                    [i.date for i in it(all_vars) if i.name == v.name])
                max_lag_endo[v] = min(
                    [i.date for i in it(all_vars) if i.name == v.name])
            except Exception as ex:
                raise SyntaxError("While parsing variable '%s': %s" % (v, ex))

        # ------------------------------------------------------------
        # arbitrary lags/leads of exogenous shocks
        subs_dict = {}
        old_var = var_ordering[:]

        for v in old_var:

            # lags
            for i in np.arange(2, abs(max_lag_endo[v])+1):
                # for lag l need to add l-1 variable
                var_l = Variable(v.name + "_LAG" + str(i-1))

                if i == 2:
                    # var_l_1 = Variable(v.name, date=-1)
                    var_l_1 = v(-1)
                else:
                    var_l_1 = Variable(v.name + "_LAG" + str(i-2), date=-1)

                subs_dict[Variable(v.name, date=-i)] = var_l(-1)
                var_ordering.append(var_l)
                equations.append(Equation(var_l, var_l_1))

            # leads
            for i in np.arange(2, abs(max_lead_endo[v])+1):
                var_l = Variable(v.name + "_LEAD" + str(i-1))

                var_l_1 = v(+1)
                # i > 2 can not be handled by method anyways
                # if i == 2:
                # var_l_1 = v(+1)
                # else:
                # var_l_1 = Variable(v.name + "_LEAD" + str(i-2), date=+1)

                subs_dict[Variable(v.name, date=+i)] = var_l(+1)
                var_ordering.append(var_l)
                equations.append(Equation(var_l, var_l_1))

        equations = [eq.subs(subs_dict) for eq in equations]

        cov = cal['covariances']

        nshock = len(shk_ordering)
        npara = len(par_ordering)

        info = {'nshock': nshock, 'npara': npara}
        QQ = sympy.zeros(nshock, nshock)

        for key, value in cov.items():
            shocks = key.split(",")

            if len(shocks) == 1:
                shocks.append(shocks[0])

            if len(shocks) == 2:
                shocki = Shock(shocks[0].strip())
                shockj = Shock(shocks[1].strip())

                indi = shk_ordering.index(shocki)
                indj = shk_ordering.index(shockj)

                QQ[indi, indj] = eval(str(value), context)
                QQ[indj, indi] = QQ[indi, indj]

        nobs = len(obs_equations)
        # HH = sympy.zeros(nobs, nobs)
        HH = zeros(nobs, nobs)

        if measurement_errors is not None:
            for key, value in cal['measurement_errors'].items():
                shocks = key.split(",")

                if len(shocks) == 1:
                    shocks.append(shocks[0])

                if len(shocks) == 2:
                    shocki = Shock(shocks[0].strip())
                    shockj = Shock(shocks[1].strip())

                    indi = measurement_errors.index(shocki)
                    indj = measurement_errors.index(shockj)

                    HH[indi, indj] = eval(value, context)
                    HH[indj, indi] = HH[indi, indj]

        context['sum'] = np.sum
        context['range'] = range
        for obs in obs_equations.items():
            obs_equations[obs[0]] = eval(obs[1], context)

        calibration = model_yaml['calibration']['parameters']

        if 'parafunc' not in cal:
            cal['parafunc'] = {}

        model_dict = {
            'var_ordering': var_ordering,
            'const_var': c_var,
            'const_eq': const_eq,
            'par_ordering': par_ordering,
            'shk_ordering': shk_ordering,
            'other_parameters': other_para,
            'other_para': other_para,
            'para_func': cal['parafunc'],
            'calibration': calibration,
            'steady_state': steady_state,
            'init_values': init_values,
            'equations': equations,
            'covariance': QQ,
            'measurement_errors': HH,
            'meas_ordering': measurement_errors,
            'info': info,
            'make_log': make_log,
            '__data__': model_yaml,
            'mod_name': dec['name'],
            'observables': observables,
            'obs_equations': obs_equations,
            'file': mtxt
        }

        model = cls(**model_dict)

        model.func_file = ffile

        model.get_matrices()

        model.par_fix = np.array(model.p0())
        p_names = [p.name for p in model.parameters]
        model.prior = model['__data__']['estimation']['prior']
        model.prior_arg = [p_names.index(pp) for pp in model.prior.keys()]
        model.prior_names = [str(pp) for pp in model.prior.keys()]
        model.observables = [str(o) for o in observables]
        model.oo = np.array(model.observables)

        return model
