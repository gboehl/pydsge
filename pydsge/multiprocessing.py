#!/bin/python
# -*- coding: utf-8 -*-
import dill
from sys import platform


def serializer_unix(func):
    """Dirty hack that transforms the non-serializable function to a serializable one (when using dill)
    ...
    Don't try that at home!
    """

    fname = func.__name__
    exec("dark_%s = func" % fname, locals(), globals())

    def vodoo(*args, **kwargs):
        return eval("dark_%s(*args, **kwargs)" % fname)

    return vodoo


def serializer(*functions):
    if platform == "darwin" or platform == "linux":
        rtn = []
        for func in functions:
            rtn.append(serializer_unix(func))
    else:
        fstr = dill.dumps(functions, recurse=True)
        rtn = dill.loads(fstr)
    if len(functions) == 1:
        return rtn[0]
    else:
        return rtn
