from pydsge import *
import pickle

#Creating the New version with deliberate difference
yaml_file, data_file = example
mod = DSGE.read(yaml_file)
par = mod.set_par('calib')
shock_list = ('e_u', 4.0, 0) # (name, size, period) ## size 3 instead of 4!!
X1, (L1, K1), _ = mod.irfs(shock_list, verbose=True)