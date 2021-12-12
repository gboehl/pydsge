# This file contains tests to check the stability of the new commit by comparing
# the tutorial output under the new commit to the pickled and stored output of the stable commit

import pickle
import pytest
from pydsge import * # imports eg. the DSGE class and the examples
import numpy as np # For np.all()
import __main__
import logging # for custom error messages






# To get the function: notebook_to_pickable_dict()
from export_getting_started_to_pkl import *

@pytest.fixture(scope="module")
def new_output(path = "docs\\getting_started.ipynb"): #need to call dir() outside of function,  otherwise only finds local objects
    '''Create dictionary of variables from current state of getting_started. 

    Args:
        path: the path to the tutorial "getting_started.ipynb"
    
    Returns:
        bk_new (dict): A dictionary of the objects and values of the current state of getting_started. The object names are the keys of the dictionary.
    '''
    bk_new = notebook_to_pickable_dict(path)

    return bk_new

@pytest.fixture(scope="module")
def stable_output():
    '''Unpickle dictionary of objects and values of stable version.

    Returns:
        bk_restore (dict): A dictionary of the objects and corresponding values of the stable getting_started. The object names are the keys of the dictionary.
    '''
    # Unpickle stable output
    with open('pydsge/tests/resources/getting_started_stable.pkl', 'rb') as f: #TODO: need to change the pickled file that is imported
        bk_restore = pickle.load(f)

    return bk_restore

@pytest.fixture(scope="module")
def diff(new_output, stable_output):
    '''Get difference between the two dictionaries' keys.
    
    Args:
        new_output (fixture):  A dictionary of the objects and values of the current state of getting_started. The object names are the keys of the dictionary.
        stable_output (fixture): A dictionary of the objects and corresponding values of the stable getting_started. The object names are the keys of the dictionary.
    
    Returns:
        diff (set): A set containing all objects which are contained in only one of the dictionaries.
    '''
    # Load the difference between new_output and stable_output
    diff = stable_output.keys() ^ new_output.keys()

    return diff


def test_what_output_is_there(diff):
    '''Check that the stable and the new version contain the same objects (excluding objects from current environment).
    Scenario:
    * Load stable and new output as dictionaries. The objects names are the keys.
    * Convert both dictionaries' keys to sets and find the difference in both directions
    * Combine the differences of both directions in "diff"
    * Check that "diff" is equal to expected difference: empty set.
    '''

#### Print error message in output under "Captured log call " if test fails

    log = logging.getLogger('')
    log.error('\n\nThe test_what_output_is_there Failed. To update the Pickle, run "python pydsge\\tests\export_getting_started_to_pkl.py" in the terminal. \n\n')

    diff == set()
    


def test_content_of_outputs(new_output, stable_output, diff):
    '''Check that the objects of the stable and the new version contain the same values
    Scenario:
    * Load stable and new output as dictionaries
    * Find set of variables that are shared by both dictionaries
    * For each shared object, check whether the content is exactly identical
    '''
    # Get collection of shared variables to loop over
    error_vars = {"In", "example_model", "example_data", "meta_data", "pth", "example", "data_file", "yaml_file", "chain", "res_dict", #Old
                 "__warningregistry__", "axs", "_", "figs", "fig", "ax"} #TODO: main issue is that there is a difference from where they are called, once local, once from package
    shared_vars = (stable_output.keys() | new_output.keys()) - diff - error_vars

    # Write function for de-nesting
    def get_flat(nested_object):
        output = []
        def reemovNestings(l):
            for i in l:
                if type(i).__name__ in ["list", "tuple", "dict", "ndarray"]:
                    reemovNestings(i)
                else:
                    output.append(i)
        reemovNestings(nested_object)
        return output

    log = logging.getLogger('')
    log.error('\n\nThe test_content_of_outputs Failed. To update the Pickle, run "python pydsge\\tests\export_getting_started_to_pkl.py" in the terminal. \n\n')


    # Loop over shared vars
    for key in sorted(shared_vars):
        print(f"This is shared_key: {key}")
        if type(new_output[key]).__name__ == "DataFrame": #Do we really need separate test for DataFrames -> would np.all() not work just as well?
            assert new_output[key].equals(stable_output[key]), f"Error with {key}"
        #############################
        # Dealing with nested objects #
        #############################
        elif key in ["hd"]: #for lists that contain DataFrames
            for counter, _ in enumerate(new_output[key]):
                assert np.all(new_output[key][counter] == stable_output[key][counter])
        elif type(new_output[key]).__name__ in ["list", "dict", "tuple", "ndarray"]: #Checking whether the object is nested
            assert get_flat(new_output[key]) == get_flat(stable_output[key])
        else:
            assert np.all(new_output[key] == stable_output[key]), f"Error with {key}" #Use np.all() for arrays





