#!/bin/python
# -*- coding: utf-8 -*-

def serializer(func):
    fname = func.__name__
    import dill
    import os
    if not os.path.exists('__pycache__'):
        os.makedirs('__pycache__')
    with open(f"__pycache__/{fname}.pkl", "wb") as f:
        dill.dump(func, f, recurse=True)

    def vodoo(*args, **kwargs):
        import dill
        with open(f"__pycache__/{fname}.pkl", "rb") as f:
            return dill.load(f)(*args, **kwargs)

    return vodoo
