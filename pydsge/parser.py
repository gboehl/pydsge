#!/bin/python
# -*- coding: utf-8 -*-

import os
import re
import yaml
import itertools
import sympy
import time
import sys
import numpy as np
import scipy.stats as sst
import scipy.optimize as sso
import cloudpickle as cpickle
from sys import platform
from copy import deepcopy
from .clsmethods import DSGE_RAW
from .symbols import Variable, Equation, Shock, Parameter, TSymbol
from sympy.matrices import Matrix, zeros


class DSGE(DSGE_RAW):
    """Base class. Every model is an instance of the DSGE class and inherents its methods."""

    def __init__(self, *kargs, **kwargs):
        super(DSGE, self).__init__(self, *kargs, **kwargs)

        fvars = []
        lvars = []

        # get forward looking variables
        for eq in self["equations"]:

            variable_too_far = [
                v for v in eq.atoms() if isinstance(v, Variable) and v.date > 1
            ]
            variable_too_early = [
                v for v in eq.atoms() if isinstance(v, Variable) and v.date < -1
            ]

            eq_fvars = [v for v in eq.atoms() if isinstance(v, TSymbol) and v.date > 0]
            eq_lvars = [v for v in eq.atoms() if isinstance(v, TSymbol) and v.date < 0]

            fvars = list(set(fvars).union(eq_fvars))
            lvars = list(set(lvars).union(set(eq_lvars)))

        self["info"]["nstate"] = len(self.variables) + len(fvars)

        self["fvars"] = fvars
        self["fvars_lagged"] = [Parameter("__LAGGED_" + f.name) for f in fvars]
        self["lvars"] = lvars
        self["re_errors"] = [Shock("eta_" + v.name) for v in self["fvars"]]

        self["re_errors_eq"] = []
        i = 0
        for fv, lag_fv in zip(fvars, self["fvars_lagged"]):
            self["re_errors_eq"].append(
                Equation(fv(-1) - lag_fv - self["re_errors"][i], 0)
            )
            i += 1

        self["perturb_eq"] = self["equations"]

        context = {}

        return

    def __repr__(self):
        return "A DSGE Model."

    @property
    def equations(self):
        return self["equations"]

    @property
    def variables(self):
        return self["var_ordering"]

    # ->
    @property
    def const_var(self):
        return self["const_var"]

    @property
    def const_eq(self):
        return self["const_eq"]

    # <-

    @property
    def parameters(self):
        return self["par_ordering"]

    @property
    def par_names(self):
        return [p.name for p in self["par_ordering"]]

    @property
    def shocks(self):
        return [str(s) for s in self["shk_ordering"]]

    @property
    def mod_name(self):
        return self["mod_name"]

    @property
    def neq(self):
        return len(self["perturb_eq"])

    @property
    def neq_fort(self):
        return self.neq + self.neta

    @property
    def neta(self):
        return len(self["fvars"])

    @property
    def nobs(self):
        return len(self["observables"])

    @property
    def neps(self):
        return len(self["shk_ordering"])

    @property
    def npara(self):
        return len(self.parameters)

    def p0(self):
        return list(map(lambda x: self["calibration"][str(x)], self.parameters))

    def get_matrices(self, matrix_format="numeric"):

        from sympy.utilities.lambdify import lambdify, implemented_function

        # check for uniqueness
        nvars = []
        for v in self["var_ordering"]:
            if v in nvars:
                print(
                    "[DSGE.read:]".ljust(15, " ")
                    + " variable `%s` is defined twice..." % v
                )
            else:
                nvars.append(v)
        self["var_ordering"] = nvars

        vlist = self["var_ordering"] + self["fvars"]
        llist = [l(-1) for l in self["var_ordering"]] + self["fvars_lagged"]

        slist = self["shk_ordering"]

        subs_dict = {}

        eq_cond = self["perturb_eq"] + self["re_errors_eq"]

        sub_var = self["var_ordering"]
        subs_dict.update({v: 0 for v in sub_var})
        subs_dict.update({v(1): 0 for v in sub_var})
        subs_dict.update({v(-1): 0 for v in sub_var})

        svar = len(vlist)
        evar = len(slist)
        rvar = len(self["re_errors"])
        ovar = len(self["observables"])

        sub_var = self["var_ordering"]
        fvarl = [l(+1) for l in sub_var]
        lvarl = [l(-1) for l in sub_var]

        no_var = len(sub_var)
        no_lvar = len(lvarl)

        bb = zeros(1, no_var + no_lvar)
        bb_PSI = zeros(1, evar)

        if self["const_var"]:
            AA = zeros(no_var - 1, no_var)
            BB = zeros(no_var - 1, no_var)
            CC = zeros(no_var - 1, no_var)
            PSI = zeros(no_var - 1, evar)

            bb_var = filter(lambda x: x.date <= 0, self["const_eq"].atoms(Variable))
            bb_fwd = [x for x in self["const_eq"].atoms(Variable) if x.date > 0]

            if bb_fwd:
                raise NotImplementedError(
                    "Forward looking variables in the constraint equation are not (yet) implemented: ",
                    *bb_fwd
                )

            full_var = sub_var + lvarl

            for v in bb_var:
                v_j = full_var.index(v)
                bb[v_j] = -(self["const_eq"]).set_eq_zero.diff(v).subs(subs_dict)

            shocks = filter(lambda x: x, self["const_eq"].atoms(Shock))

            for s in shocks:
                s_j = slist.index(s)
                bb_PSI[s_j] = -(self["const_eq"]).set_eq_zero.diff(s).subs(subs_dict)

        else:
            AA = zeros(no_var, no_var)
            BB = zeros(no_var, no_var)
            CC = zeros(no_var, no_var)
            PSI = zeros(no_var, evar)

        eq_i = 0
        for eq in self["perturb_eq"]:

            A_var = filter(lambda x: x.date > 0, eq.atoms(Variable))
            for v in A_var:
                v_j = fvarl.index(v)
                AA[eq_i, v_j] = (eq).set_eq_zero.diff(v).subs(subs_dict)

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

        ZZ0 = zeros(ovar, no_var)
        ZZ1 = zeros(ovar, 1)

        eq_i = 0
        for obs in self["observables"]:
            eq = self["obs_equations"][str(obs)]
            ZZ1[eq_i, 0] = eq.subs(subs_dict)

            curr_var = filter(lambda x: x.date >= 0, eq.atoms(Variable))

            for v in curr_var:
                v_j = vlist.index(v)
                ZZ0[eq_i, v_j] = eq.diff(v).subs(subs_dict)

                if self.const_var is v:
                    self.const_obs = obs

            eq_i += 1

        from collections import OrderedDict

        context = dict([(p.name, p) for p in self.parameters])
        sol_dict = {}

        def ctf_reducer(f):
            """hack to reduce the calls-to-function"""

            def reducer(*x):
                try:
                    return sol_dict[f, x]
                except KeyError:
                    res = f(*x)
                    sol_dict[f, x] = res
                    return res

            return reducer

        # standard functions
        context["exp"] = implemented_function("exp", np.exp)
        context["log"] = implemented_function("log", np.log)
        context["sqrt"] = implemented_function("sqrt", np.sqrt)

        # distributions
        context["normpdf"] = implemented_function("normpdf", ctf_reducer(sst.norm.pdf))
        context["normcdf"] = implemented_function("normcdf", ctf_reducer(sst.norm.cdf))
        context["normppf"] = implemented_function("normppf", ctf_reducer(sst.norm.ppf))
        context["norminv"] = implemented_function("norminv", ctf_reducer(sst.norm.ppf))

        # things defined in *_funcs.py
        if self.func_file and os.path.exists(self.func_file):
            import importlib.util as iu
            import inspect

            spec = iu.spec_from_file_location("module", self.func_file)
            module = iu.module_from_spec(spec)
            spec.loader.exec_module(module)

            funcs_list = [
                o for o in inspect.getmembers(module) if inspect.isroutine(o[1])
            ]

            for func in funcs_list:
                context[func[0]] = implemented_function(func[0], ctf_reducer(func[1]))

        ss = {}
        checker = np.zeros_like(self["other_para"], dtype=bool)
        suc_loop = True

        while ~checker.all():

            # print(checker)
            # raise if loop was unsuccessful
            raise_error = not suc_loop
            suc_loop = False  # set to check if any progress in loop
            for i, p in enumerate(self["other_para"]):
                if not checker[i]:
                    try:
                        ss[str(p)] = eval(str(self["para_func"][p.name]), context)
                        context[str(p)] = ss[str(p)]
                        checker[i] = True
                        suc_loop = True  # loop was successful
                    except NameError as error:
                        if raise_error:
                            error_msg = str(error)
                            if not os.path.exists(self.func_file):
                                fname = os.path.basename(self.func_file)
                                error_msg += (
                                    " (info: a file named `%s` was not found)" % fname
                                )
                            raise type(error)(
                                str(error)
                                + " (are definitions in `para_func` circular?)"
                            ).with_traceback(sys.exc_info()[2])

        ZZ0 = lambdify([self.parameters + self["other_para"]], ZZ0)
        ZZ1 = lambdify([self.parameters + self["other_para"]], ZZ1)

        PSI = lambdify([self.parameters + self["other_para"]], PSI)

        AA = lambdify([self.parameters + self["other_para"]], AA)
        BB = lambdify([self.parameters + self["other_para"]], BB)
        CC = lambdify([self.parameters + self["other_para"]], CC)
        bb = lambdify([self.parameters + self["other_para"]], bb)
        bb_PSI = lambdify([self.parameters + self["other_para"]], bb_PSI)

        psi = lambdify(
            [self.parameters], [ss[str(px)] for px in self["other_para"]]
        )  # , modules=context_f)

        def compile(px):
            return list(px) + psi(list(px))

        self.pcompile = compile
        self.parafunc = [p.name for p in self["other_para"]], psi
        self.psi = psi
        self.PSI = PSI

        self.ZZ0 = ZZ0
        self.ZZ1 = ZZ1
        self.AA = AA
        self.BB = BB
        self.CC = CC
        self.bb = bb
        self.bb_PSI = bb_PSI

        QQ = lambdify([self.parameters + self["other_para"]], self["covariance"])
        HH = lambdify(
            [self.parameters + self["other_para"]], self["measurement_errors"]
        )

        self.QQ = QQ
        self.HH = HH

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

        if "processed_raw_model" in globals():

            use_cached = True
            func_file = mfile[:-5] + "_funcs.py"

            if os.path.exists(func_file):
                ff = open(func_file)
                ftxt = ff.read()
                use_cached &= processed_raw_model.fdict.get("ffile_raw") == ftxt

            use_cached &= processed_raw_model.fdict["yaml_raw"] == mtxt

        if use_cached:
            pmodel = deepcopy(processed_raw_model)

        else:

            func_file = mfile[:-5] + "_funcs.py"

            pmodel = cls.parse(mtxt, func_file)

            pmodel.fdict = {}
            pmodel.fdict["yaml_raw"] = mtxt

            if os.path.exists(func_file):
                ff = open(func_file)
                ftxt = ff.read()
                pmodel.fdict["ffile_raw"] = ftxt

            pmodel_dump = cpickle.dumps(pmodel, protocol=4)
            pmodel.fdict["model_dump"] = pmodel_dump
            pmodel.name = pmodel.mod_name
            pmodel.path = os.path.dirname(mfile)
            pmodel.debug = platform == "darwin" or platform == "win32"
            if pmodel.debug:
                print(
                    "[DSGE:]".ljust(15, " ")
                    + " Parallelization disabled under Windows and Mac due to a problem with pickling some of the symbolic elements. Sorry..."
                )

            processed_raw_model = deepcopy(pmodel)

        if verbose:
            duration = np.round(time.time() - st, 3)
            if duration < 0.01:
                duration = "the speed of light"
            else:
                str(duration) + "s"
            print(
                "[DSGE:]".ljust(15, " ") + " Reading and parsing done in %s." % duration
            )

        return pmodel

    @classmethod
    def load(cls, npzfile, force_parse=False, verbose=False):

        global processed_raw_model

        if verbose:
            st = time.time()

        fdict = dict(np.load(npzfile, allow_pickle=True))

        mtxt = str(fdict["yaml_raw"])

        try:
            if force_parse:
                raise Exception
            pmodel = cpickle.loads(fdict["model_dump"])
            # pickling errors across protocols are likely
            # so this is a test:
            pmodel_dump = cpickle.dumps(pmodel, protocol=4)
        except:
            use_cached = False

            if "processed_raw_model" in globals():
                use_cached = processed_raw_model.fdict["yaml_raw"] == mtxt

            if use_cached:
                pmodel = deepcopy(processed_raw_model)
            else:
                import tempfile

                try:
                    # cumbersome: load the text of the *_funcs file, write it to a temporary file, just to use it as a module
                    ftxt = str(fdict["ffile_raw"])
                    tfile = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
                    tfile.write(ftxt)
                    tfile.close()
                    ffile = tfile.name
                except KeyError:
                    ffile = ""

                pmodel = cls.parse(mtxt, ffile)
                pmodel_dump = cpickle.dumps(pmodel, protocol=4)

                try:
                    tfile.close()
                    os.unlink(tfile.name)
                except:
                    pass

        pmodel.fdict = fdict
        pmodel.name = str(fdict["name"])
        pmodel.path = os.path.dirname(npzfile)
        pmodel.data = cpickle.loads(fdict["data"])
        pmodel.fdict["model_dump"] = pmodel_dump

        pmodel.debug = platform == "darwin" or platform == "win32"
        if pmodel.debug:
            print(
                "[DSGE:]".ljust(15, " ")
                + " Parallelization disabled under Windows and Mac due to a problem with pickling some of the symbolic elements. Sorry..."
            )

        for ob in pmodel.fdict.keys():
            if str(pmodel.fdict[ob]) == "None":
                pmodel.fdict[ob] = None

        if verbose:
            print(
                "[DSGE:]".ljust(15, " ")
                + " Loading and parsing done in %ss." % np.round(time.time() - st, 5)
            )

        return pmodel

    @classmethod
    def parse(cls, mtxt, ffile):
        """ """

        mtxt = mtxt.replace("^", "**")
        mtxt = mtxt.replace(";", "")
        mtxt = re.sub(r"@ ?\n", " ", mtxt)
        model_yaml = yaml.safe_load(mtxt)

        dec = model_yaml["declarations"]
        cal = model_yaml["calibration"]

        name = dec["name"]

        var_ordering = [Variable(v) for v in dec["variables"]]
        par_ordering = [Parameter(v) for v in cal["parameters"]]
        shk_ordering = [Shock(v) for v in cal["covariances"]]

        if "parafunc" in cal:
            other_para = [Parameter(v) for v in cal["parafunc"]]
        else:
            other_para = []

        context = [
            (s.name, s) for s in var_ordering + par_ordering + shk_ordering + other_para
        ]
        context = dict(context)

        if "observables" in model_yaml["equations"]:
            observables = [Variable(v) for v in model_yaml["equations"]["observables"]]
            obs_equations = model_yaml["equations"]["observables"]
        else:
            observables = []
            obs_equations = dict()

        if "constrained" in dec:
            c_var = Variable(dec["constrained"][0])
            # only one constraint allowed
            raw_const = model_yaml["equations"]["constraint"][0]

            if "=" in raw_const:
                lhs, rhs = str.split(raw_const, "=")
            else:
                lhs, rhs = raw_const, "0"

            try:
                lhs = eval(lhs, context)
                rhs = eval(rhs, context)
            except TypeError as e:
                raise SyntaxError(
                    "While parsing %s, got this error: %s" % (raw_const, repr(e))
                )

            const_eq = Equation(lhs, rhs)
        else:
            c_var = []
            const_eq = dict()

        if "measurement_errors" in dec:
            measurement_errors = [Shock(v) for v in dec["measurement_errors"]]
        else:
            measurement_errors = None

        if "make_log" in dec:
            make_log = [Variable(v) for v in dec["make_log"]]
        else:
            make_log = []

        steady_state = [0]
        init_values = [0]

        context["__builtins__"] = None

        equations = []

        if "model" in model_yaml["equations"]:
            raw_equations = model_yaml["equations"]["model"]
        else:
            raw_equations = model_yaml["equations"]

        if len(raw_equations) + 1 != len(var_ordering):
            raise SyntaxError(
                "I got %s variables but %s equations"
                % (len(var_ordering), len(raw_equations) + 1)
            )

        for eq in raw_equations:
            if "=" in eq:
                lhs, rhs = str.split(eq, "=")
            else:
                lhs, rhs = eq, "0"

            try:
                lhs = eval(lhs, context)
                rhs = eval(rhs, context)
            except TypeError as e:
                raise SyntaxError(
                    "While parsing %s, got this error: %s" % (eq, repr(e))
                )

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
                    [i.date for i in it(all_shocks) if i.name == s.name]
                )
                max_lag_exo[s] = min(
                    [i.date for i in it(all_shocks) if i.name == s.name]
                )
            except Exception as ex:
                raise SyntaxError("While parsing shock '%s': %s" % (s, ex))

        # arbitrary lags of exogenous shocks
        for s in shk_ordering:
            if abs(max_lag_exo[s]) > 0:
                var_s = Variable(s.name + "_VAR")
                var_ordering.append(var_s)
                equations.append(Equation(var_s, s))

                subs1 = [s(-i) for i in np.arange(1, abs(max_lag_exo[s]) + 1)]
                subs2 = [var_s(-i) for i in np.arange(1, abs(max_lag_exo[s]) + 1)]
                subs_dict = dict(zip(subs1, subs2))
                equations = [eq.subs(subs_dict) for eq in equations]

        all_vars = [list(eq.atoms(Variable)) for eq in equations]
        max_lead_endo = dict.fromkeys(var_ordering)
        max_lag_endo = dict.fromkeys(var_ordering)

        for v in var_ordering:
            try:
                max_lead_endo[v] = max(
                    [i.date for i in it(all_vars) if i.name == v.name]
                )
                max_lag_endo[v] = min(
                    [i.date for i in it(all_vars) if i.name == v.name]
                )
            except Exception as ex:
                raise SyntaxError("While parsing variable '%s': %s" % (v, ex))

        # ------------------------------------------------------------
        # arbitrary lags/leads of exogenous shocks
        subs_dict = {}
        old_var = var_ordering[:]

        for v in old_var:

            # lags
            for i in np.arange(2, abs(max_lag_endo[v]) + 1):
                # for lag l need to add l-1 variable
                var_l = Variable(v.name + "_LAG" + str(i - 1))

                if i == 2:
                    # var_l_1 = Variable(v.name, date=-1)
                    var_l_1 = v(-1)
                else:
                    var_l_1 = Variable(v.name + "_LAG" + str(i - 2), date=-1)

                subs_dict[Variable(v.name, date=-i)] = var_l(-1)
                var_ordering.append(var_l)
                equations.append(Equation(var_l, var_l_1))

            # leads
            for i in np.arange(2, abs(max_lead_endo[v]) + 1):
                var_l = Variable(v.name + "_LEAD" + str(i - 1))

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

        cov = cal["covariances"]

        nshock = len(shk_ordering)
        npara = len(par_ordering)

        info = {"nshock": nshock, "npara": npara}
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
        HH = zeros(nobs, nobs)

        if measurement_errors is not None:
            for key, value in cal["measurement_errors"].items():
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

        context["sum"] = np.sum
        context["range"] = range
        for obs in obs_equations.items():
            obs_equations[obs[0]] = eval(obs[1], context)

        calibration = model_yaml["calibration"]["parameters"]

        if "parafunc" not in cal:
            cal["parafunc"] = {}

        model_dict = {
            "var_ordering": var_ordering,
            "const_var": c_var,
            "const_eq": const_eq,
            "par_ordering": par_ordering,
            "shk_ordering": shk_ordering,
            "other_parameters": other_para,
            "other_para": other_para,
            "para_func": cal["parafunc"],
            "calibration": calibration,
            "steady_state": steady_state,
            "init_values": init_values,
            "equations": equations,
            "covariance": QQ,
            "measurement_errors": HH,
            "meas_ordering": measurement_errors,
            "info": info,
            "make_log": make_log,
            "__data__": model_yaml,
            "mod_name": dec["name"],
            "observables": observables,
            "obs_equations": obs_equations,
            "file": mtxt,
        }

        model = cls(**model_dict)

        model.func_file = ffile

        model.get_matrices()

        model.par_fix = np.array(model.p0())
        p_names = [p.name for p in model.parameters]
        model.prior = model["__data__"]["estimation"]["prior"]
        model.prior_arg = [p_names.index(pp) for pp in model.prior.keys()]
        model.prior_names = [str(pp) for pp in model.prior.keys()]
        model.observables = [str(o) for o in observables]
        model.vo = np.array(model.observables)
        model.ve = np.array(model.shocks)

        return model
