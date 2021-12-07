import nbformat
import pickle
from nbconvert import PythonExporter


def nbconvert_python(path):
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


def is_picklable(obj):
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


def filter_pickable(global_vars):
    """Filter the variables that are pickable

    Args:
        global_vars (array-like): The names of variables to get
    
    Returns:
        dict: Dictionary containing names of objects and their values
    
    """
    bk = {}
    for k in global_vars:
        obj = global_vars[k]
        if is_picklable(obj):
            try:
                bk.update({k: obj})
            except TypeError:
                pass
    return bk


def notebook_to_pickable_dict(path):
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
    # Step 1: Convert notebook to script
    code = nbconvert_python(path)
    code = code.replace("get_ipython()", "# get_ipython()")

    # Step 2: Execute script and save variables in dictionary
    d= {}
    exec(code, d)
    d.pop("__builtins__")

    # Step 3: Filter for pickable variables
    bk = filter_pickable(d)
    return bk


def save_to_pkl(path, obj):
    """Save object to pickle file.
    
    Args:
        path (str): Path to save pickle file
        obj : Object to be saved
        
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    # excute jupyter notebook and save global variables
    notebook_path = "docs\\getting_started.ipynb"

    bk = notebook_to_pickable_dict(notebook_path)

    # to save session
    save_path = "pydsge/tests/resources/getting_started_stable.pkl"
    save_to_pkl(save_path, bk)


if __name__ == "__main__":
    main()
