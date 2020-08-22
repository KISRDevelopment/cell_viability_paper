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
import utils.make_cfgs

# create Model Combinations
# with open("cfgs/models/yeast_gi_full_model.json", "r") as f:
#     base_cfg = json.load(f)
#utils.make_gi_model_combs.main(base_cfg, "../tmp/model_cfgs/yeast_gi")

#models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/yeast_gi", 
#    "../results/task_yeast_gi_hybrid", 8, n_runs=40, exclude=lambda s: 'smf' not in s)

# models.cv.main("models.null_model", "cfgs/models/yeast_gi_full_model.json", "../results/task_yeast_gi_hybrid/null")
# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", "../results/task_yeast_gi_hybrid/null_scrambled", scramble=True, 
#     num_processes=1)

# utils.make_cfgs.main("cfgs/models/sweep_yeast_gi_pairwise_cfgs.json", "../tmp/model_cfgs/yeast_gi_pairwisesweep")
# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/yeast_gi_pairwisesweep", 
#     "../results/task_yeast_gi_hybrid", 10, n_runs=40)

#utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_gi/topology~go~pairwise~smf.json", "topology", "../tmp/model_cfgs/gi_topology_sweep")
# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/gi_topology_sweep", 
#     "../results/task_yeast_gi_hybrid", 15, n_runs=40)

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_hybrid/topology--lid~go~pairwise--spl~smf", 
#     num_processes=10)

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid/mn", 
#     num_processes=20)

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid/orm", 
#     num_processes=20, type="orm")

# models.cv.main("models.null_model", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid/null", 
#     num_processes=10)

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", 
#     "../results/task_yeast_gi_hybrid/null_scrambled", 
#     num_processes=10,
#     scramble=True)

#if not os.path.exists('../results/yeast_gi_hybrid_figures'):
 #   os.makedirs('../results/yeast_gi_hybrid_figures')

#analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid", "../results/yeast_gi_hybrid_figures/model_comp.xlsx")

costanzo_task_path = "../generated-data/task_yeast_gi_costanzo"
costanzo_targets_path = "../generated-data/targets/task_yeast_gi_costanzo_bin_simple.npz"
costanzo_splits_path = "../generated-data/splits/task_yeast_gi_costanzo_10reps_4folds_0.20valid.npz"

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", 
#     "../results/task_yeast_gi_costanzo/full", 
#     num_processes = 10,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )


# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_costanzo/topology--lid~go~pairwise--spl~smf", 
#     num_processes = 20,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_costanzo/mn", 
#     num_processes=20,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path)

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_costanzo/orm", 
#     num_processes=20, type="orm",
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path)

# models.cv.main("models.null_model", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_costanzo/null", 
#     num_processes=10,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path)

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", 
#     "../results/task_yeast_gi_costanzo/null_scrambled", 
#     num_processes=10,
#     scramble=True,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path)