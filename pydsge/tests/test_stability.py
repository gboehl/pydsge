"""This file contains tests to check the stability of the new commit."""
import pytest
import numpy as np  # for dealing with numpy pickle
import pandas as pd  # for dealing with data frames
import logging  # for custom error messages
from pathlib import Path  # for windows-Unix compatibility

from export_getting_started_to_pkl import notebook_exec_result_flattened
from export_getting_started_to_pkl import to_ndarray


@pytest.fixture(scope="module")
def new_output(path="docs/getting_started.ipynb"):
    """Create dictionary of variables from current state of getting_started.

    Args:
        path: the path to the tutorial "getting_started.ipynb"

    Returns:
        bk_new (dict): A dictionary of the objects and values of the current state
         of getting_started. The object names are the keys of the dictionary.
    """
    # Convert notebook to script, execute and filter-out environment vars
    bk_new = notebook_exec_result_flattened(path)
    # Convert dictionary entries to np arrays
    bk_new_array = to_ndarray(bk_new)

    return bk_new_array


@pytest.fixture(scope="module")
def stable_output():
    """Unpickle dictionary of objects and values of stable version.

    Returns:
        bk_restore (dict): A dictionary of the objects and corresponding values of the
        stable getting_started. The object names are the keys of the dictionary.
    """
    path_to_pickle = Path("pydsge/tests/resources/getting_started_stable.npz")
    # Unpickle stable output
    with open(path_to_pickle, "rb") as f:
        npzfile = np.load(f, allow_pickle=False)
        bk_restore = dict(npzfile)

    return bk_restore


@pytest.fixture(scope="module")
def diff(new_output, stable_output):
    """Get difference between the two dictionaries' keys.

    Args:
        new_output (fixture):  A dictionary of the objects and values of
        the current state of getting_started. The object names are the
        keys of the dictionary.
        stable_output (fixture): A dictionary of the objects and corresponding values
        of the stable getting_started. The object names are the keys of the dictionary.

    Returns:
        diff (set): A set containing all objects which are contained in only one of the
        dictionaries.
    """
    # Load the difference between new_output and stable_output
    return stable_output.keys() ^ new_output.keys()


@pytest.mark.regression()
def test_what_output_is_there(diff):
    """Check that the object names are identical.

    Compare the names of the objects from the stable and the new version.
    Scenario:
    * Load stable and new output as dictionaries.
    The objects names are the dictionary keys.
    * Convert both dictionaries' keys to sets and find the difference
    in both directions
    * Combine the differences of both directions in "diff"
    * Check that "diff" is equal to expected difference: empty set.
    """
    # Print error message in report (under "Captured log call ") if test fails
    log = logging.getLogger("")
    log.error(
        r"\n\nThe test_what_output_is_there Failed. "
        r"To update the Pickle, run "
        r'"python pydsge/tests/export_getting_started_to_pkl.py" '
        r"in the terminal. Or allow CI bot to uptade it. \n\n"
    )

    assert diff == set()


# TODO: parametrize  on different tutorials
@pytest.mark.regression()
def test_content_of_outputs(new_output, stable_output, diff, atol=1e-07):
    """Check that objects contain the same values.

    Compare the values of the objects from the stable and the new version.
    Scenario:
    * Load stable and new output as dictionaries
    * Find set of variables that are shared by both dictionaries
    * For each shared object, check whether the content is exactly identical
    """
    # Get collection of shared variables to loop over
    error_vars = set()
    shared_vars = (stable_output.keys() | new_output.keys()) - diff - error_vars

    # Write function for de-nesting
    def get_flat(nested_object):
        """Apply function recursively until lowest level is reached.

        The exception are DataFrames.
        Returns: DataFrame, non-iterable object or array with non-iterable objects
        """
        if isinstance(nested_object, pd.DataFrame) or not np.iterable(nested_object):
            return nested_object
        output = []

        def remove_nestings(nested_object):
            for i in nested_object:
                if type(i).__name__ in ["list", "tuple", "dict", "ndarray"]:
                    remove_nestings(i)
                else:
                    output.append(i)

        remove_nestings(nested_object)
        return output

    # Print error message in report (under "Captured log call ") if test fails
    log = logging.getLogger("")
    log.error(
        r"\n\nThe test_content_of_outputs Failed. "
        r"To update the Pickle, run "
        r'"python pydsge/tests/export_getting_started_to_pkl.py" '
        r"in the terminal. Or allow CI bot to uptade it. \n\n"
    )

    # Loop over shared vars
    for key in sorted(shared_vars):
        # De-nest nested objects, except for pd.DataFrames
        stable = get_flat(stable_output[key])
        new = get_flat(new_output[key])

        # For efficiency check DataFrames separately
        if type(new_output[key]).__name__ == "DataFrame":
            pd.testing.assert_frame_equal(new, stable, atol=atol), f"Error with {key}"

        # Comparison for arrays, strings, floats and ints (i.e. all others)
        elif type(new_output[key]).__name__ in ["string", "float", "int", "ndarray"]:
            # Check if dealing with numpy string
            if isinstance(new_output[key], np.ndarray) and new_output[
                key
            ].dtype not in (float, int):
                assert new == stable, f"Error with {key}"
            else:
                np.testing.assert_allclose(
                    new, stable, atol=atol, equal_nan=True
                ), f"Error with {key}"

        # NOTE: This should not occur
        else:
            raise AssertionError(
                f"Error, the type of de-nestde key: {key} is "
                f"{type(new_output[key]).__name__}"
            )
