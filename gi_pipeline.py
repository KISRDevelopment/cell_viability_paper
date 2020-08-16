import os 
import subprocess 
import sys 
import json 
import shlex

import utils.make_smf_single_feature_sweeps
import utils.make_gi_model_combs
import models.multiple_cv
import models.cv 
import analysis.tbl_model_comp

#create Model Combinations
# with open("cfgs/models/yeast_gi_full_model.json", "r") as f:
#     base_cfg = json.load(f)
# utils.make_gi_model_combs.main(base_cfg, "../tmp/model_cfgs/yeast_gi")

# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/yeast_gi", 
#     "../results/task_yeast_gi_hybrid", 10, n_runs=40)
# models.cv.main("models.null_model", "cfgs/models/yeast_gi_full_model.json", "../results/task_yeast_gi_hybrid/null")
# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", "../results/task_yeast_gi_hybrid/null_scrambled", scramble=True, 
#     num_processes=1)
# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/yeast_gi_pairwisesweep", 
#     "../results/task_yeast_gi_hybrid", 10, n_runs=40)

if not os.path.exists('../results/yeast_gi_hybrid_figures'):
    os.makedirs('../results/yeast_gi_hybrid_figures')

analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid", "../results/yeast_gi_hybrid_figures/model_comp.xlsx")
