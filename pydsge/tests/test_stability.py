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


def is_picklable(obj): #TODO: import function from file where getting_started.py is pickled (currently getting it from toy_getting_started)
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
    #TODO: create docstring
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
    #TODO: create docstring
    print(os.getcwd())
    with open('pydsge/tests/resources/toy_example.pkl', 'rb') as f: #TODO: need to change the pickled file that is imported
        bk_restore = pickle.load(f)

    return bk_restore

def test_what_output_is_there(new_output, stable_output):
    '''Check whether bk lists are identical [TODO: order of loading will
    influence the difference, decide for something]
    '''
    # Get difference between the two dictionaries
    diff = (set(stable_output.keys()) - set(new_output.keys())) | (set(new_output.keys()) - set(stable_output.keys()))
    print(diff)

    assert diff == {"bk", "In", "Out"} #[order of loading will influence the difference]

def test_content_of_outputs(new_output, stable_output):
    #TODO: create docstring
    # Get difference between the two dictionaries
    diff = (set(stable_output.keys()) - set(new_output.keys())) | (set(new_output.keys()) - set(stable_output.keys()))

    # Get collection of shared variables to loop over
    error_vars = {"In", "example_model", "example_data", "meta_data", "pth", "example", "data_file", "yaml_file", "chain", "res_dict"} #TODO: main issue is that there is a difference from where they are called, once local, once from package
    shared_vars = (set(stable_output.keys()) | set(new_output.keys())) - diff - error_vars #TODO: here getting manually rid of "In", but should do that elsewhere

    # Loop over shared vars
    for key in shared_vars:
        if type(new_output[key]).__name__ == "DataFrame": #Do we really need separate test for DataFrames -> would np.all() not work just as well?
            assert new_output[key].equals(stable_output[key]), f"Error with {key}"
        else:
            assert np.all(new_output[key] == stable_output[key]), f"Error with {key}" #Use np.all() for arrays
