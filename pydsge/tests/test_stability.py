"""This file contains tests to check the stability of the new commit."""
import pytest
import numpy as np  # for dealing with numpy pickle
import logging  # for custom error messages

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
    bk_new = notebook_exec_result_flattened(path)
    bk_new_array = to_ndarray(bk_new)

    return bk_new_array


@pytest.fixture(scope="module")
def stable_output():
    """Unpickle dictionary of objects and values of stable version.

    Returns:
        bk_restore (dict): A dictionary of the objects and corresponding values of the
        stable getting_started. The object names are the keys of the dictionary.
    """
    # Unpickle stable output
    with open("pydsge/tests/resources/getting_started_stable.npz", "rb") as f:
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
    diff = stable_output.keys() ^ new_output.keys()

    return diff


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
    # Print error message in output under "Captured log call " if test fails
    log = logging.getLogger("")
    log.error(
        r"\n\nThe test_what_output_is_there Failed."
        r" To update the Pickle, run"
        r' "python pydsge/tests/export_getting_started_to_pkl.py"'
        r" in the terminal. \n\n"
    )

    assert diff == set()


def nan_undecidable(array):
    """Filter out objects where numpy cannot decide nans."""
    return (
        issubclass(np.asarray(array).dtype.type, str)
        or np.asarray(array).dtype.hasobject
    )


def array_equal(a1, a2):
    """Let numpy handle nan where possible."""
    if nan_undecidable(a1) or nan_undecidable(a2):
        return np.array_equal(a1, a2, equal_nan=False)
    else:
        return np.allclose(a1, a2, equal_nan=True)


# @pytest.mark.parametrize("")
# TODO: don't parametrize  on inputs, but for different tutorials??
@pytest.mark.regression()
def test_content_of_outputs(new_output, stable_output, diff):
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
        """Apply function recursively until lowest level is reached."""
        if not np.iterable(nested_object):
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

    log = logging.getLogger("")
    log.error(
        r"\n\nThe test_content_of_outputs Failed."
        r"To update the Pickle, run"
        r'"python pydsge/tests/export_getting_started_to_pkl.py"'
        r"in the terminal. \n\n"
    )

    # Loop over shared vars
    for key in sorted(shared_vars):
        if type(new_output[key]).__name__ == "DataFrame":
            assert new_output[key].equals(stable_output[key]), f"Error with {key}"
        # for nested objects
        elif key in ["hd"]:  # for lists that contain DataFrames
            for counter, _ in enumerate(new_output[key]):
                assert array_equal(
                    new_output[key][counter], stable_output[key][counter]
                ), f"Error with hd {counter}"
        elif type(new_output[key]).__name__ in [
            "list",
            "dict",
            "tuple",
            "ndarray",
        ]:  # Checking whether the object is nested
            stable = get_flat(stable_output[key])
            new = get_flat(new_output[key])
            assert array_equal(new, stable), f"Error with {key}"
        else:
            assert array_equal(new_output[key], stable_output[key]), f"Error with {key}"
