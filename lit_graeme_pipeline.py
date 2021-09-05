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
    task_path="../generated-data/task_yeast_gi_hybrid",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
    target_col = "is_not_negative")

models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/lit_graeme2019/mn", 
    num_processes=20,
    task_path="../generated-data/task_yeast_gi_hybrid",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
    target_col = "is_not_negative")


