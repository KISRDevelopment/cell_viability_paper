import numpy as np 
import pandas as pd 
import os 
import models.cv 
import feature_preprocessing.shared_sgo_count

# generate the pairedwise shared sGO terms feature
#feature_preprocessing.shared_sgo_count.main("../generated-data/ppc_yeast")

# run the SLANT model
models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_slant.json", 
    "../results/lit_graeme2019/slant", 
    num_processes=20,
    target_col = "is_neutral")

