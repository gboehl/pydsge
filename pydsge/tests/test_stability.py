# This file contains tests to check the stability of the new commit by comparing
# the tutorial output under the new commit to the pickled and stored output of the stable commit

import pickle
import pytest
from pydsge import * # imports eg. the DSGE class and the examples

# Load output under new commit
from resources.getting_started import * #TODO: difference test will fail at the moment

@pytest.fixture("new_output")
def get_output_newcommit():
    #TODO: create docstring
    def is_picklable(obj): #TODO: import function from file whether getting_started.py is pickled
        """
        check if obj can be dumped to a .pkl file
        """
        try:
            pickle.dumps(obj)
        except Exception:
            return False
        return True

    ## Get relevant variables in environment
    bk_new = {}
    for k in dir():
        obj = globals()[k]
        if not k.startswith('_') and is_picklable(obj):
            try:
                bk_new.update({k: obj})
            except TypeError:
                pass

    return bk_new

@pytest.fixture("stable_output")
def get_output_stable():
    #TODO: create docstring
    with open('resources/toy_example.pkl', 'rb') as f: #TODO: need to change the pickled file that is imported
        bk_restore = pickle.load(f)

    return bk_restore

def test_what_output_is_there(new_output, stable_output):
    '''Check whether bk lists are identical [TODO: order of loading will
    influence the difference, decide for something]
    '''
    # Get difference between the two dictionaries
    diff = (set(stable_output.keys()) - set(new_output.keys())) | (set(new_output.keys()) - set(stable_output.keys()))
    assert diff == {"bk", "bk_new", "bk_restore"} #[order of loading will influence the difference]

def test_content_of_outputs(new_output, stable_output):
    #TODO: create docstring
    # Get difference between the two dictionaries
    diff = (set(stable_output.keys()) - set(new_output.keys())) | (set(new_output.keys()) - set(stable_output.keys()))

    # Get collection of shared variables to loop over
    shared_vars = (set(bk_restore.keys()) | set(bk_new.keys())) - diff - {"In"} #TODO: here getting manually rid of "In", but should do that elsewhere

    # Loop over shared vars
    for key in shared_vars:
        if type(bk_new[key]).__name__ == "DataFrame": #Do we really need separate test for DataFrames -> would np.all() not work just as well?
            assert bk_new[key].equals(bk_restore[key]), f"Error with {key}"
        else:
            assert np.all(bk_new[key] == bk_restore[key]), f"Error with {key}" #Use np.all() for arrays
