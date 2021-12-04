import nbformat
import pickle
from nbconvert import PythonExporter

def _nbconvert_python(path):
    """
    convert jupyter notebook to python code
    """
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    body, _ = PythonExporter().from_notebook_node(nb)
    return body


def _is_picklable(obj):
    """
    check if obj can be dumped to a .pkl file
    """
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


def _get_global_vars(global_vars):
    bk = {}
    for k in global_vars:
        obj = globals()[k]
        if _is_picklable(obj):
            try:
                bk.update({k: obj})
            except TypeError:
                pass
    return bk


def backup_notebook_global_vars(path):
    '''
    excute jupyter notebook and save global variables
    '''
    code = _nbconvert_python(path)
    code = code.replace("get_ipython()", "# get_ipython()")

    _var_before_exec = dir()
    exec(code)

    _global_vars = set(dir()) - set(_var_before_exec)
    _global_vars = list(filter(lambda k : not k.startswith('_'), _global_vars))
    print(f"global_vars:{_global_vars}")

    bk = _get_global_vars(_global_vars)
    return bk


def save_to_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# excute jupyter notebook and save global variables
notebook_path = "docs\\getting_started.ipynb"

code = _nbconvert_python(notebook_path)
code = code.replace("get_ipython()", "# get_ipython()")

_var_before_exec = dir()
exec(code)

_global_vars = set(dir()) - set(_var_before_exec)
_global_vars = list(filter(lambda k : not k.startswith('_'), _global_vars))
print(f"global_vars:{_global_vars}")

bk = _get_global_vars(_global_vars)

# to save session
save_path = "pydsge\\tests\\resources\\getting_started_stable.pkl"
save_to_pkl(save_path, bk)
