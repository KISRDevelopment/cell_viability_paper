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

"""
    Create Model Combinations
"""
# with open("cfgs/models/yeast_gi_full_model.json", "r") as f:
#     base_cfg = json.load(f)
# utils.make_gi_model_combs.main(base_cfg, "../tmp/model_cfgs/yeast_gi")

# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/yeast_gi", 
#    "../results/task_yeast_gi_hybrid", 8, n_runs=40, exclude=lambda s: 'smf' not in s)

# models.cv.main("models.null_model", "cfgs/models/yeast_gi_full_model.json", "../results/task_yeast_gi_hybrid/null")
# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", "../results/task_yeast_gi_hybrid/null_scrambled", scramble=True, 
#     num_processes=1)

# utils.make_cfgs.main("cfgs/models/sweep_yeast_gi_pairwise_cfgs.json", "../tmp/model_cfgs/yeast_gi_pairwisesweep")
# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/yeast_gi_pairwisesweep", 
#     "../results/task_yeast_gi_hybrid", 10, n_runs=40)

# utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_gi/topology~go~pairwise~smf.json", "topology", "../tmp/model_cfgs/gi_topology_sweep")
# models.multiple_cv.main("models.gi_nn", "../tmp/model_cfgs/gi_topology_sweep", 
#     "../results/task_yeast_gi_hybrid", 15, n_runs=40)

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_hybrid/refined", 
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

# if not os.path.exists('../results/yeast_gi_hybrid_figures'):
#    os.makedirs('../results/yeast_gi_hybrid_figures')

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_hybrid/refined_no_sgo", 
#     remove_specs=["go"],
#     num_processes=10)

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid/mn_no_sgo", 
#     remove_specs=["sgo"],
#     num_processes=20)

# analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid", "../results/yeast_gi_hybrid_figures/model_comp.xlsx")

"""
    Costanzo data
"""

# costanzo_task_path = "../generated-data/task_yeast_gi_costanzo"
# costanzo_targets_path = "../generated-data/targets/task_yeast_gi_costanzo_bin_simple.npz"
# costanzo_splits_path = "../generated-data/splits/task_yeast_gi_costanzo_10reps_4folds_0.20valid.npz"

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", 
#     "../results/task_yeast_gi_costanzo/full", 
#     num_processes = 10,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
#     "../results/task_yeast_gi_costanzo/refined", 
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

"""
    Definitive GI Dataset
"""
costanzo_task_path = "../generated-data/task_yeast_gi"
costanzo_targets_path = "../generated-data/targets/task_yeast_gi_bin_interacting.npz"
costanzo_splits_path = "../generated-data/splits/task_yeast_gi_10reps_4folds_0.20valid.npz"

# models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_full_model.json", 
#     "../results/task_yeast_gi/full", 
#     num_processes = 10,
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path
# )

models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_refined_model.json", 
    "../results/task_yeast_gi/refined", 
    num_processes = 20,
    task_path = costanzo_task_path,
    targets_path = costanzo_targets_path,
    splits_path = costanzo_splits_path
)

models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/task_yeast_gi/mn", 
    num_processes=20,
    task_path = costanzo_task_path,
    targets_path = costanzo_targets_path,
    splits_path = costanzo_splits_path)

exit(0)

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

models.cv.main("models.gi_nn", "cfgs/models/pombe_gi_refined_model.json", 
    "../results/task_pombe_gi_binary/refined_no_sgo_and_smf", 
    num_processes = 20,
    targets_path=targets_path,
    remove_specs=["go", "smf"]
)

models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
    "../results/task_pombe_gi_binary/mn_no_sgo_and_smf", 
    num_processes = 20,
    remove_specs=["sgo", "smf"],
    targets_path=targets_path
)

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

models.cv.main("models.gi_nn", "cfgs/models/human_gi_refined_model.json", 
    "../results/task_human_gi/refined_no_sgo_and_smf", 
    remove_specs=["go","smf"],
    num_processes = 20
)

models.cv.main("models.gi_mn", "cfgs/models/human_gi_mn.json", 
    "../results/task_human_gi/mn_no_sgo_and_smf",
    remove_specs=["sgo","smf"], 
    num_processes = 20
)

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

models.cv.main("models.gi_nn", "cfgs/models/dro_gi_refined_model.json", 
    "../results/task_dro_gi/refined_no_sgo_and_smf", 
    num_processes = 20,
    remove_specs=["go","smf"],
)

models.cv.main("models.gi_mn", "cfgs/models/dro_gi_mn.json", 
    "../results/task_dro_gi/mn_no_sgo_and_smf", 
    num_processes = 20,
    remove_specs=["sgo","smf"]
)
