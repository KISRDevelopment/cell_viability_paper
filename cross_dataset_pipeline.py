import numpy as np 
import pandas as pd 
import tasks.merge_costanzo_hybrid
import utils.bin_interacting
import utils.cv_gi
import models.gi_mn 
import json 
import models.cv 
import utils.bin_simple

def load_cfg(path, **kwargs):
    with open(path, 'r') as f:
        cfg = json.load(f)

    cfg.update(kwargs)

    return cfg 

gi_task_path = "../generated-data/task_yeast_gi_merged"
costanzo_targets = "../generated-data/targets/task_yeast_gi_merged_costanzo_bin_interacting.npz"
hybrid_targets = "../generated-data/targets/task_yeast_gi_merged_hybrid_bin_interacting.npz"
cv_path = "../generated-data/splits/task_yeast_gi_merged_10reps_4folds_0.20valid.npz"

# tasks.merge_costanzo_hybrid.main("../generated-data/ppc_yeast", 
#     "../generated-data/task_yeast_gi_costanzo", "../generated-data/task_yeast_gi_hybrid", 
#     gi_task_path)
#utils.bin_interacting.main(gi_task_path,  "costanzo_bin")
#utils.bin_interacting.main(gi_task_path, "hybrid_bin")
# utils.bin_simple.main(gi_task_path,  "costanzo_bin")
# utils.bin_simple.main(gi_task_path, "hybrid_bin")
#utils.cv_gi.main(gi_task_path, 10, 4, 0.2)

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_merged/mn_costanzo_4way", 
#     num_processes = 20,
#     targets_path="../generated-data/targets/task_yeast_gi_merged_costanzo_bin_simple.npz",
#     task_path=gi_task_path,
#     splits_path=cv_path,
#     trained_model_path="../tmp/yeast_gi_merged/mn_costanzo",
#     add_repfold_to_trained_model_path=True
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_merged/mn_costanzo", 
#     num_processes = 20,
#     targets_path=costanzo_targets,
#     task_path=gi_task_path,
#     splits_path=cv_path,
#     trained_model_path="../tmp/yeast_gi_merged/mn_costanzo",
#     add_repfold_to_trained_model_path=True
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_merged/mn_costanzo_to_hybrid", 
#     num_processes = 20,
#     targets_path=hybrid_targets,
#     task_path=gi_task_path,
#     splits_path=cv_path,
#     train_model=False,
#     trained_model_path="../tmp/yeast_gi_merged/mn_costanzo",
#     add_repfold_to_trained_model_path=True
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_merged/mn_hybrid", 
#     num_processes = 20,
#     targets_path=hybrid_targets,
#     task_path=gi_task_path,
#     splits_path=cv_path,
#     trained_model_path="../tmp/yeast_gi_merged/mn_hybrid",
#     add_repfold_to_trained_model_path=True
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_merged/mn_hybrid_to_costanzo", 
#     num_processes = 20,
#     targets_path=costanzo_targets,
#     task_path=gi_task_path,
#     splits_path=cv_path,
#     train_model=False,
#     trained_model_path="../tmp/yeast_gi_merged/mn_hybrid",
#     add_repfold_to_trained_model_path=True
# )

#
# Full Costanzo Merger
#

gi_task_path = "../generated-data/task_yeast_gi_merged_all"
# tasks.merge_costanzo_hybrid.main("../generated-data/ppc_yeast", 
#     "../generated-data/task_yeast_gi_costanzo_all", "../generated-data/task_yeast_gi_hybrid", 
#     gi_task_path)
# utils.bin_interacting.main(gi_task_path,  "costanzo_bin")
# utils.bin_interacting.main(gi_task_path, "hybrid_bin")
# utils.bin_simple.main(gi_task_path,  "costanzo_bin")
# utils.bin_simple.main(gi_task_path, "hybrid_bin")
# utils.cv_gi.main(gi_task_path, 10, 4, 0.2)

costanzo_targets = "../generated-data/targets/task_yeast_gi_merged_all_costanzo_bin_interacting.npz"
hybrid_targets = "../generated-data/targets/task_yeast_gi_merged_all_hybrid_bin_interacting.npz"
cv_path = "../generated-data/splits/task_yeast_gi_merged_all_10reps_4folds_0.20valid.npz"

models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/task_yeast_gi_merged_all/mn_costanzo", 
    num_processes = 20,
    targets_path=costanzo_targets,
    task_path=gi_task_path,
    splits_path=cv_path,
    trained_model_path="../tmp/yeast_gi_merged_all/mn_costanzo",
    add_repfold_to_trained_model_path=True
)
