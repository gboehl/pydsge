import sympy
from sympy.printing.str import StrPrinter

from sympy.core.cache import clear_cache

clear_cache()


StrPrinter._print_TSymbol = lambda self, x: x.__str__()


class Parameter(sympy.Symbol):

    def __init__(self, name, exp_date=0):
        super(Parameter, self).__init__()
        self.name = name
    # def __new__(self, name, exp_date=0):
    #     super(Parameter,self).__new__( name)

    def __repr__(self):
        return(self.name)

    def __set_prior(self, prior):
        self.prior = prior


class TSymbol(sympy.Symbol):

    #__xnew_cached_ = staticmethod(sympy.Symbol.__new_stage2__)

    def __init__(self, name, **args):
        super(TSymbol, self).__init__()

        if 'date' not in args:
            self._assumptions['date'] = 0
            self.assumptions0['date'] = 0
        else:
            self._assumptions['date'] = args['date']
            self.assumptions0['date'] = args['date']
        if 'exp_date' not in args:
            self._assumptions['exp_date'] = 0
            self.assumptions0['exp_date'] = 0
        else:
            self._assumptions['exp_date'] = args['exp_date']
            self.assumptions0['exp_date'] = args['exp_date']

        self._mhash = None
        self.__hash__()

        return None

    def __call__(self, lead):
        newdate = int(self.date) + int(lead)
        newname = str(self.name)
        # print 'creating', newname, newdate
        clear_cache()
        return self.__class__(newname, date=newdate)

    @property
    def date(self):
        return self.assumptions0['date']

    @property
    def exp_date(self):
        return self.assumptions0['exp_date']

    def _hashable_content(self):
        return (self.name, str(self.date), str(self.exp_date))

    def __getstate__(self):
        return {
            # 'date': self.date,
            # 'name': self.name,
            # 'exp_date': self.exp_date,
            # 'is_commutative': self.is_commutative,
            # '_mhash': self._mhash
        }

    def class_key(self):
        return (2, 0, self.name, self.date)

    @property
    def lag(self):
        return self.date

    def __str__(self):
        if self.lag == 0:
            result = self.name
        else:
            result = self.name + r"(" + str(self.lag) + r")"
        return result

    # def __repr__(self):
    #     return self.__str__()

    # def _repr_(self):
    #     return self.__str__()

    # def repr(self):
    #     return self.__str__()

    # def _print_TSymbol(self):
    #     return self.__str__()

    # def _print(self):
    #     return self.__str__()


class Variable(TSymbol):

    @property
    def fortind(self):
        if self.date <= 0:
            return('v_'+self.name)
        else:
            return('v_E'+self.name)

    def __str__(self):
        if self.exp_date == 0:
            result = super(Variable, self).__str__()
        else:
            result = 'E[' + str(self.exp_date) + ']' + \
                super(Variable, self).__str__()

        return result

    def __repr__(self):
        return self.__str__()

    __sstr__ = __str__


class LaggedExpectation(Variable):

    def __init__(self, name, date=0, exp_date=0):
        Variable.__init__(self, name, date)
        self.exp_date = exp_date

    def __getstate_(self):
        return {
            'date': self.date,
            'name': self.name,
            'is_commutative': self.is_commutative,
            '_mhash': self._mhash
        }

    def _hashable_content(self):
        return (self.name, self.date, self.lag)

    def __str__(self):
        """
        """
        if self.lag == 0:
            result = "E_{t-j}[self.name]"
        else:
            pass


class Shock(TSymbol):

    @property
    def fortind(self):
        if self.date <= 0:
            return('e_'+self.name)
        else:
            return('e_E'+self.name)


class Equation(sympy.Equality):

    # def __init__(self, lhs, rhs, name=None):
    #    super(sympy.Equality, self).__init__(lhs, rhs)
    def __new__(cls, lhs, rhs, name=None):
        return super(sympy.Equality, cls).__new__(cls, lhs, rhs)

    @property
    def set_eq_zero(self):
        return(self.lhs - self.rhs)

    @property
    def variables(self):
        l = [v for v in self.atoms() if isinstance(v, Variable)]
        return l
