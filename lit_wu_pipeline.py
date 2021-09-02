import numpy as np 
import pandas as pd 
import os 
import utils.cv_simple_repeated_random
import models.cv 
import tasks.yeast_gi_wu2014
import utils.make_biogrid_dataset

if not os.path.exists('../generated-data/lit_tasks'):
    os.makedirs('../generated-data/lit_tasks')
if not os.path.exists('../generated-data/lit_splits'):
    os.makedirs('../generated-data/lit_splits')

# make biogrid dataset similar to the one used by Wu 2014
utils.make_biogrid_dataset.main(4932, 0, "../generated-data/biogrid_old_yeast_sl")

# make GI task according to Wu's methodology
tasks.yeast_gi_wu2014.main("../generated-data/ppc_yeast", "../generated-data/biogrid_old_yeast_sl", "../generated-data/lit_tasks/wu2014")

# Split it in 2/3 training 1/3 testing
utils.cv_simple_repeated_random.main("../generated-data/lit_tasks/wu2014", 50, 3, 0.2, "../generated-data/lit_splits/wu2014")

# Execute MN model
models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/lit_wu2014/mn", 
    num_processes=20,
    target_col = "bin",
    task_path="../generated-data/lit_tasks/wu2014",
    splits_path="../generated-data/lit_splits/wu2014.npz")


