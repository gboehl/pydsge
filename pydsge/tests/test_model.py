# This file contains tests for the solving of the model

from pydsge import *
import numpy as np
import pytest

@pytest.fixture(scope="module")
def parametrized_minimal_nk():
    # Retrieve the example data
    yaml_file = "pydsge/examples/minimal_nk.yaml"

    #Read-in the yaml_file
    mod = DSGE.read(yaml_file)

    # Load in parameters ("calibrate")
    _ = mod.set_par('calib')
    # Get parameters as dictionary
    par1, par2 = mod.get_par(asdict=True)

    return mod, par1, par2


def test_minimal_nkmodel(parametrized_minimal_nk, tolerance = 6):
    """
    Scenario: Load the minimal_nk.yaml model and compare TODO: <What?> to analytically derived solution.
    """
    # Get model
    mod, par1, par2 = parametrized_minimal_nk

    # Simulate demand shock
    shock_list = ('e_u', 4.0, 0) # (name, size, period)
    X1, (_, _), _ = mod.irfs(shock_list, verbose=False)

    #get estimated impules
    infl_modImp,  outgap_modImp = X1["Pi"].iloc[0], X1["y"].iloc[0]

    #get analytically derived demand shock
    ## - From method of undetermined coefficients verified linear solution for Pi and y
    gamma_1 = (1-par1["beta"]*par1["rho_u"])/par2["kappa"]
    gamma_2 = (par1["phi_pi"]+par1["rho_u"])/(par1["rho_u"]-1)
    theta_pi = 1/((gamma_1-gamma_2)*(par1["rho_u"]-1))
    theta_y = gamma_2 * theta_pi + 1/(par1["rho_u"]-1)

    infl_analyImp = theta_pi*shock_list[1] #inflation as linear response to shock
    outgap_analyImp = theta_y*shock_list[1] #output gap as linear response to shock

    # Check model solution against analytical solution
    np.testing.assert_almost_equal(infl_modImp, infl_analyImp, decimal = tolerance)
    np.testing.assert_almost_equal(outgap_modImp, outgap_analyImp, decimal = tolerance)


