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

def load_cfg(path, **kwargs):
    with open(path, 'r') as f:
        cfg = json.load(f)
    cfg.update(kwargs)
    return cfg 

"""
    Feature Selection on Yeast Hybrid  GI Dataset
"""
# with open("cfgs/models/yeast_gi_full_model.json", "r") as f:
#     base_cfg = json.load(f)
# utils.make_gi_model_combs.main(base_cfg, "../tmp/model_cfgs/yeast_gi")

# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/yeast_gi", 
#    "../results/task_yeast_gi_hybrid_fs", n_processors=8, n_runs=40)

# # analyze all models on training set
# analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid_fs", 
#     "../tmp/yeast_gi_model_comp.xlsx", analysis.tbl_model_comp.GI_LABELS)

# # the top performing model has only 4 feature sets : pairwise, topology, smf, and sgo

# # sweep the pairwise sets
# utils.make_cfgs.main("cfgs/models/sweep_yeast_gi_pairwise_cfgs.json", "../tmp/model_cfgs/yeast_gi_pairwisesweep")
# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/yeast_gi_pairwisesweep", "../results/task_yeast_gi_hybrid_fs", 10, n_runs=40)

# # sweep the topology
# utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_gi/topology~go~pairwise~smf.json", "topology", "../tmp/model_cfgs/gi_topology_sweep")
# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/gi_topology_sweep", 
#     "../results/task_yeast_gi_hybrid_fs", 15, n_runs=40)

# # analyze all models on training set
# analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid_fs", 
#      "../tmp/yeast_gi_model_comp.xlsx", analysis.tbl_model_comp.GI_LABELS)

# # we pick LID, SPL, sGO and SMF for the refined and MN and OR models
# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_hybrid_fs/refined", 
#     num_processes=20)

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_fs/mn", 
#     num_processes=20)

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_fs/orm", 
#     num_processes=20, type="orm")

# models.cv.main("models.null_model", "cfgs/models/yeast_gi_full_model.json", "../results/task_yeast_gi_hybrid_fs/null")

# # final report
# analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid_fs", 
#      "../tmp/yeast_gi_model_comp.xlsx", analysis.tbl_model_comp.GI_LABELS)


# test!
models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", 
    "../results/task_yeast_gi_hybrid_train_test/full", 
    num_processes=1,
    n_runs=1,
    verbose=True,
    task_path="../generated-data/task_yeast_gi_hybrid_train_test",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_train_test.npz")


models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
    "../results/task_yeast_gi_hybrid_train_test/refined", 
    num_processes=1,
    n_runs=1,
    verbose=True,
    task_path="../generated-data/task_yeast_gi_hybrid_train_test",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_train_test.npz")

models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/task_yeast_gi_hybrid_train_test/mn", 
    num_processes=1,
    n_runs=1,
    verbose=True,
    task_path="../generated-data/task_yeast_gi_hybrid_train_test",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_train_test.npz")

models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/task_yeast_gi_hybrid_train_test/orm", 
    num_processes=1,
    n_runs=1,
    verbose=True,
    type="orm",
    task_path="../generated-data/task_yeast_gi_hybrid_train_test",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_train_test.npz")

models.cv.main("models.null_model", "cfgs/models/yeast_gi_full_model.json", 
    "../results/task_yeast_gi_hybrid_train_test/null", 
    num_processes=1,
    n_runs=1,
    verbose=True,
    task_path="../generated-data/task_yeast_gi_hybrid_train_test",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_train_test.npz")


"""
    Costanzo data
"""

costanzo_task_path = "../generated-data/task_yeast_gi_costanzo_asym"
costanzo_targets_path = "../generated-data/targets/task_yeast_gi_costanzo_asym_bin_simple.npz"
costanzo_splits_path = "../generated-data/splits/task_yeast_gi_costanzo_asym_10reps_4folds_0.20valid.npz"

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", 
#     "../results/task_yeast_gi_costanzo/full", 
#     num_processes = 10,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_costanzo_asym/refined", 
#     num_processes = 20,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_costanzo_asym_binary/refined", 
#     num_processes = 20,
#     task_path = costanzo_task_path,
#     targets_path = "../generated-data/targets/task_yeast_gi_costanzo_asym_bin_interacting.npz",
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

"""
    Definitive GI Dataset
"""
costanzo_task_path = "../generated-data/task_yeast_gi_thres15"
costanzo_targets_path = "../generated-data/targets/task_yeast_gi_thres15_bin_interacting.npz"
costanzo_splits_path = "../generated-data/splits/task_yeast_gi_thres15_10reps_4folds_0.20valid.npz"

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", 
#     "../results/task_yeast_gi/full", 
#     num_processes = 10,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_thres15_binary/refined", 
#     num_processes = 20,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi/mn", 
#     num_processes=20,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path)

"""
    Yeast Binary
"""

# targets_path = "../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz"

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_hybrid_binary/refined", 
#     num_processes = 20,
#     targets_path = targets_path
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_binary/mn", 
#     num_processes=20,
#     targets_path = targets_path)

# models.cv.main("models.null_model", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_binary/null", 
#     num_processes=20,
#     targets_path = targets_path)

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_hybrid_binary/null_scrambled", 
#     num_processes=20,
#     scramble=True,
#     targets_path = targets_path)

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_hybrid_binary/refined_no_sgo", 
#     num_processes = 20,
#     remove_specs=["go"],
#     targets_path = targets_path
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_binary/mn_no_sgo", 
#     num_processes=20,
#     remove_specs=["sgo"],
#     targets_path = targets_path)

"""
    Yeast Negative vs. All
"""
targets_path = "../generated-data/targets/task_yeast_gi_hybrid_bin_negative.npz"

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_hybrid_binary_negative/refined", 
#     num_processes = 20,
#     targets_path = targets_path
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_binary_negative/mn", 
#     num_processes=20,
#     targets_path = targets_path)

# models.cv.main("models.null_model", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_binary_negative/null", 
#     num_processes=20,
#     targets_path = targets_path)


# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_slant.json", 
#     "../results/task_yeast_gi_hybrid_binary_negative/slant", 
#     num_processes=20,
#     targets_path = targets_path)


# """
#     Yeast Costanzo Binary
# """
# costanzo_task_path = "../generated-data/task_yeast_gi_costanzo"
# costanzo_targets_path = "../generated-data/targets/task_yeast_gi_costanzo_bin_interacting.npz"
# costanzo_splits_path = "../generated-data/splits/task_yeast_gi_costanzo_10reps_4folds_0.20valid.npz"

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", 
#     "../results/task_yeast_gi_costanzo_binary/full", 
#     num_processes = 10,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_costanzo_binary/refined", 
#     num_processes = 20,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_costanzo_binary/mn", 
#     num_processes=20,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path)

# models.cv.main("models.null_model", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_costanzo_binary/null", 
#     num_processes=10,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path)

"""
    Pombe
"""

# models.cv.main("models.gi_nn", "cfgs/models/pombe_gi_refined_model.json", 
#     "../results/task_pombe_gi/refined", 
#     num_processes = 20
# )

# models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
#     "../results/task_pombe_gi/mn", 
#     num_processes = 20
# )

# models.cv.main("models.null_model", "cfgs/models/pombe_gi_mn.json", 
#     "../results/task_pombe_gi/null", 
#     num_processes=20)

# models.cv.main("models.gi_nn", "cfgs/models/pombe_gi_refined_model.json", 
#     "../results/task_pombe_gi/refined_no_sgo", 
#     remove_specs=["go"],
#     num_processes = 20
# )

# models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
#     "../results/task_pombe_gi/mn_no_sgo", 
#     remove_specs=["sgo"],
#     num_processes = 20
# )

# """
#     Pombe Binary
# """

targets_path = "../generated-data/targets/task_pombe_gi_bin_interacting.npz"
# # models.cv.main("models.gi_nn", "cfgs/models/pombe_gi_refined_model.json", 
# #     "../results/task_pombe_gi_binary/refined", 
# #     num_processes = 20,
# #     targets_path=targets_path
# # )

# # models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
# #     "../results/task_pombe_gi_binary/mn", 
# #     num_processes = 20,
# #     targets_path=targets_path
# # )

# # models.cv.main("models.null_model", "cfgs/models/pombe_gi_mn.json", 
# #     "../results/task_pombe_gi_binary/null", 
# #     num_processes=20,
# #     targets_path=targets_path)

# models.cv.main("models.gi_nn", "cfgs/models/pombe_gi_refined_model.json", 
#     "../results/task_pombe_gi_binary/refined_no_sgo", 
#     num_processes = 20,
#     targets_path=targets_path,
#     remove_specs=["go"]
# )

# models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
#     "../results/task_pombe_gi_binary/mn_no_sgo", 
#     num_processes = 20,
#     remove_specs=["sgo"],
#     targets_path=targets_path
# )

# models.cv.main("models.gi_nn", "cfgs/models/pombe_gi_refined_model.json", 
#     "../results/task_pombe_gi_binary/refined_no_sgo_and_smf", 
#     num_processes = 20,
#     targets_path=targets_path,
#     remove_specs=["go", "smf"]
# )

# models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
#     "../results/task_pombe_gi_binary/mn_no_sgo_and_smf", 
#     num_processes = 20,
#     remove_specs=["sgo", "smf"],
#     targets_path=targets_path
# )

# # """
# # Human
# # """

# # models.cv.main("models.gi_nn", "cfgs/models/human_gi_refined_model.json", 
# #     "../results/task_human_gi/refined", 
# #     num_processes = 20
# # )

# # models.cv.main("models.gi_mn", "cfgs/models/human_gi_mn.json", 
# #     "../results/task_human_gi/mn", 
# #     num_processes = 20
# # )

# # models.cv.main("models.null_model", "cfgs/models/human_gi_mn.json", 
# #     "../results/task_human_gi/null", 
# #     num_processes=20)

# models.cv.main("models.gi_nn", "cfgs/models/human_gi_refined_model.json", 
#     "../results/task_human_gi/refined_no_sgo", 
#     remove_specs=["go"],
#     num_processes = 20
# )

# models.cv.main("models.gi_mn", "cfgs/models/human_gi_mn.json", 
#     "../results/task_human_gi/mn_no_sgo",
#     remove_specs=["sgo"], 
#     num_processes = 20
# )

# models.cv.main("models.gi_nn", "cfgs/models/human_gi_refined_model.json", 
#     "../results/task_human_gi/refined_no_sgo_and_smf", 
#     remove_specs=["go","smf"],
#     num_processes = 20
# )

# models.cv.main("models.gi_mn", "cfgs/models/human_gi_mn.json", 
#     "../results/task_human_gi/mn_no_sgo_and_smf",
#     remove_specs=["sgo","smf"], 
#     num_processes = 20
# )

# # """
# # Dro
# # """

# # models.cv.main("models.gi_nn", "cfgs/models/dro_gi_refined_model.json", 
# #     "../results/task_dro_gi/refined", 
# #     num_processes = 20
# # )

# # models.cv.main("models.gi_mn", "cfgs/models/dro_gi_mn.json", 
# #     "../results/task_dro_gi/mn", 
# #     num_processes = 20
# # )

# # models.cv.main("models.null_model", "cfgs/models/dro_gi_mn.json", 
# #     "../results/task_dro_gi/null", 
# #     num_processes=20)

# models.cv.main("models.gi_nn", "cfgs/models/dro_gi_refined_model.json", 
#     "../results/task_dro_gi/refined_no_sgo", 
#     num_processes = 20,
#     remove_specs=["go"],
# )

# models.cv.main("models.gi_mn", "cfgs/models/dro_gi_mn.json", 
#     "../results/task_dro_gi/mn_no_sgo", 
#     num_processes = 20,
#     remove_specs=["sgo"]
# )

# models.cv.main("models.gi_nn", "cfgs/models/dro_gi_refined_model.json", 
#     "../results/task_dro_gi/refined_no_sgo_and_smf", 
#     num_processes = 20,
#     remove_specs=["go","smf"],
# )

# models.cv.main("models.gi_mn", "cfgs/models/dro_gi_mn.json", 
#     "../results/task_dro_gi/mn_no_sgo_and_smf", 
#     num_processes = 20,
#     remove_specs=["sgo","smf"]
# )
