# check whether the  parameters values in the model are the same as in yaml_file.

from pydsge import *
import yaml as yaml
import numpy as np


yalm_file, _ = example
model= DSGE.read(yalm_file)

# get par values from the model
dsgepar= model.parameters
dsge_parvalues= model.get_par()
dsge_parvalues = np.transpose(dsge_parvalues).reshape(dsge_parvalues.size, 1)

# get the par values from yalm file
yfile= open(r"c:\Users\findh\My Drive\Bonn__MSc\5th\pydsge_OSE_Project_Fork\pydsge\examples\dfi.yaml")
mtxt=yfile.read()
mtxt=mtxt.replace(" ~ "," - " )
parsedfile= yaml.load(mtxt, Loader= yaml.FullLoader)
calib_1= parsedfile.get('calibration')
dict_parvalues= calib_1.get('parameters').values()
yamlloaded_parvalues=[]
for i in dict_parvalues:
    yamlloaded_parvalues.append([i])

# make a test
def test_calib_par():
    np.testing.assert_almost_equal(dsge_parvalues, yamlloaded_parvalues)

