import os 
import subprocess 
import sys 
import json 
import shlex

import utils.bin_simple
import utils.cv_gi 
import tasks.yeast_gi_costanzo
import tasks.yeast_gi_hybrid
import utils.make_gi_model_combs
import models.cv
import models.multiple_cv

gpath = "../generated-data/ppc_yeast"
task_path = "../generated-data/task_yeast_hybrid"
cfg_path = "cfgs/models/yeast_gi_full_model.json"

# # create task
# tasks.yeast_gi_hybrid.main(gpath, task_path)

# # create targets
# utils.bin_simple.main(task_path)

# # create splits
# utils.cv_gi.main(task_path, 10, 4, 0.2)

# create combination 
# with open(cfg_path, 'r') as f:
#     base_cfg = json.load(f)
# utils.make_gi_model_combs.main(base_cfg, "../results/yeast_gi_hybrid")

# run combinations


# run refined model
models.cv.main("models.gi_model", "cfgs/models/yeast_gi_refined_model.json", "../results/task_yeast_gi_hybrid/refined",
    num_processes=4)

