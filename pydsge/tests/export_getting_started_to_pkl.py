import nbformat
import pickle
from nbconvert import PythonExporter


def _nbconvert_python(path):
    """Use nbconvert to convert jupyter notebook to python code.
    Return the string of python code. You can then excute it with `exec()`.

    Args:
        path (str): Path of jupyter notebook

    Returns:
        str: The string of python code converted from notebook
    
    """
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    body, _ = PythonExporter().from_notebook_node(nb)
    return body


def _is_picklable(obj):
    """Check if an obj can be dumped into a pickle file.

    Args:
        obj : The Object to be judged
    
    Returns:
        bool: The result if the input can be picklable

    """
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


def _get_global_vars(global_vars):
    """Get variables from globals by names of variables.

    Args:
        global_vars (array-like): The names of variables to get
    
    Returns:
        dict: Dictionary containing names of variables and variables
    
    """
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
    """Excute jupyter notebook and then save variables defined in notebook.
    This function converts notebook to python code and then excutes the code.
    Finally it put all public variables that defined in notebook into dictionary 
    and return it.
    Parameters
    ----------
    path : str
        Path of jupyter notebook
    Returns
    -------
    bk : :dict
        Dictionary containing names of variables and variables that defined in notebook.
    """
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
    """Save object to pickle file.
    
    Args:
        path (str): Path to save pickle file
        obj : Object to be saved
        
    """
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
