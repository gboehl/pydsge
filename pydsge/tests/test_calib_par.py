# check whether the  parameters values in the model are the same as in yaml_file.

from pydsge import *
import yaml as yaml
import numpy as nps


yalm_file, data_file= example
model= DSGE.read(yalm_file)

# get par values from the model
dsgepar= model.parameters
dsge_parvalues= model.get_par()
dsge_parvalues = np.transpose(dsge_parvalues).reshape(dsge_parvalues.size, 1)

# get the par values from yalm file
yfile= open(r"pydsge/examples/dfi.yaml")
mtxt = yfile.read()
mtxt = mtxt.replace("   ~ ", "   - ")
parsedfile = yaml.safe_load(mtxt)
# parsedfile= yaml.load(yfile, Loader= yaml.FullLoader)
w11= parsedfile.get('calibration')
dict_parvalues= w11.get('parameters').values()
len(dict_parvalues)
yamlloaded_parvalues=[]
for i in dict_parvalues:
    yamlloaded_parvalues.append([i])

# make a test
def test_calib_par():
    np.testing.assert_almost_equal(dsge_parvalues, yamlloaded_parvalues)

