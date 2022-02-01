"""This file contains tests for the parsing of the model."""
import numpy as np

from pydsge import DSGE
from pydsge import example


def test_parsing_yaml():
    """Check wether yaml file is parsed correctly.

    Scenario: Load pydsge example and compare to list of parameters.
    """
    # Retrieve the example data
    yaml_file, _ = example

    # Read-in the yaml_file
    mod = DSGE.read(yaml_file)
    par = mod.set_par("calib")

    yaml_parameters = [
        "beta",
        "sigma",
        "theta",
        "phi_y",
        "phi_pi",
        "rho_z",
        "rho_u",
        "rho_r",
        "rho",
        "sig_z",
        "sig_u",
        "sig_r",
        "nub",
        "psi",
        "elb_level",
        "y_mean",
        "pi_mean",
    ]  # To-do function: get_from_yaml

    assert isinstance(par, np.ndarray)
    assert par.size == yaml_parameters.__len__()
