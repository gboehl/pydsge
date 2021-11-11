# This file contains tests for the parsing of the model
from pydsge import *
import numpy as np


def test_parsing_yaml():
    """
    Scenario: Check whether example model generated from yaml fulfills certain requirements
    """
    # Retrieve the example data
    yaml_file, _ = example

    #Read-in the yaml_file
    mod = DSGE.read(yaml_file)
    par = mod.set_par('calib')

    yaml_parameters = ["beta", "sigma", "theta", "phi_y", "phi_pi", "rho_z", "rho_u", "rho_r", "rho", "sig_z", "sig_u", "sig_r", "nub", "psi", "elb_level", "y_mean", "pi_mean"] # To-do function: get_from_yaml

    assert isinstance(par, np.ndarray)
    assert par.size == yaml_parameters.__len__()



