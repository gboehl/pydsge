# This file contains tests to check the stability of the new commit by comparing
# the tutorial output under the new commit to the pickled and stored output of the stable commit

import pickle
import pytest
from pydsge import * # imports eg. the DSGE class and the examples
import numpy as np # For np.all()
import __main__

# Load output under new commit
# from resources.getting_started import * #TODO: difference test will fail at the moment
from resources.toy_getting_started import * #TODO: delete
#from export_getting_started_to_pkl import _is_picklable

def is_picklable(obj):
    """
    check if obj can be dumped to a .pkl file
    """
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True

__main__.is_picklable = is_picklable #add is_picklable to the envirnoment for pickle load (from https://www.py4u.net/discuss/246734)



@pytest.fixture(scope="module")
def new_output(global_vars = dir()): #need to call dir() outside of function,  otherwise only finds local objects
    '''Create dictionary of variables from current state of getting_started. This function uses ALL objects that are in the current environment, so correct working critically relies on
    loading stable_output seperately.
    Note: we need to call dir() outside of function,  otherwise only finds local objects

    Args:
        global_vars: all objects of the current environment, making sure stable output is not included
    
    Returns:
        bk_new (dict): A dictionary of the objects and values of the current state of getting_started. The object names are the keys of the dictionary.
    '''
    ## Get relevant variables in environment
    bk_new = {}
    for k in global_vars:
        obj = globals()[k]
        if not k.startswith('_') and is_picklable(obj):
            try:
                bk_new.update({k: obj})
            except TypeError:
                pass

    return bk_new

@pytest.fixture(scope="module")
def stable_output():
    '''Unpickle dictionary of objects and values of stable version.

    Returns:
        bk_restore (dict): A dictionary of the objects and corresponding values of the stable getting_started. The object names are the keys of the dictionary.
    '''
    # Unpickle stable output
    with open('pydsge/tests/resources/toy_example.pkl', 'rb') as f: #TODO: need to change the pickled file that is imported
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
    # Load the difference in both directions
    diff_new2stable = set(stable_output.keys()) - set(new_output.keys())
    diff_stable2new = set(new_output.keys()) - set(stable_output.keys())
    # Combine both differences
    diff = diff_new2stable | diff_stable2new

    return diff

def test_what_output_is_there(diff):
    '''Check that the stable and the new version contain the same objects (excluding objects from current environment).
    Scenario:
    * Load stable and new output as dictionaries. The objects names are the keys.
    * Convert both dictionaries' keys to sets and find the difference in both directions
    * Combine the differences of both directions in "diff"
    * Check that "diff" is equal to expected differences: "bk", "In", "Out" are not part of the pickled output.
    '''
    assert diff == {"bk", "In", "Out"} #[order of loading will influence the difference]

def test_content_of_outputs(new_output, stable_output, diff):
    '''Check that the objects of the stable and the new version contain the same values
    Scenario:
    * Load stable and new output as dictionaries
    * Find set of variables that are shared by both dictionaries
    * For each shard object, check whether the content is exactly identical
    '''
    # Get collection of shared variables to loop over
    error_vars = {"In", "example_model", "example_data", "meta_data", "pth", "example", "data_file", "yaml_file", "chain", "res_dict"} #TODO: main issue is that there is a difference from where they are called, once local, once from package
    shared_vars = (set(stable_output.keys()) | set(new_output.keys())) - diff - error_vars 

    # Loop over shared vars
    for key in shared_vars:
        if type(new_output[key]).__name__ == "DataFrame": #Do we really need separate test for DataFrames -> would np.all() not work just as well?
            assert new_output[key].equals(stable_output[key]), f"Error with {key}"
        else:
            assert np.all(new_output[key] == stable_output[key]), f"Error with {key}" #Use np.all() for arrays
