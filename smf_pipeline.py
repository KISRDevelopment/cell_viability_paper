import os 
import subprocess 
import sys 
import json 
import shlex

import utils.make_smf_single_feature_sweeps
import utils.make_smf_model_combs
import models.multiple_cv
import models.cv 
import analysis.tbl_model_comp

#
# Yeast SMF Model Selection
#

# # create all possible model combinations
# utils.make_smf_model_combs.main("cfgs/models/yeast_smf_full_model.json", 
#     comb_output_path="../results/yeast_smf_30_fs",
#     cfg_output_path="../tmp/model_cfgs/yeast_smf"
# )

# # evaluate on training set via cross validation
# models.multiple_cv.main("models.smf_nn", 
#     "../tmp/model_cfgs/yeast_smf", 
#     "../results/yeast_smf_30_fs", n_processors=20, n_runs=50)

# # we identify SGO+Redundancy+Topology as the model with the fewest features that achieves >= 95%
# # of max performace. So we sweep Redundancy and topology features to identify which one is responsible
# # for model performance
# utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_smf/topology~go~redundancy.json", "redundancy", "../tmp/model_cfgs/yeast_smf")
# utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_smf/topology~go~redundancy.json", "topology", "../tmp/model_cfgs/yeast_smf")
# models.multiple_cv.main("models.smf_nn", 
#     "../tmp/model_cfgs/yeast_smf", 
#     "../results/yeast_smf_30_fs", n_processors=20, n_runs=50)
# models.cv.main("models.smf_nn", 
#     "cfgs/models/yeast_smf_refined_model.json", 
#     "../results/yeast_smf_30_fs/refined",
#     num_processes=20)

# we further constrain the refined model to be a simple OR model
#models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", "../results/yeast_smf_30_fs/orm", num_processes=20)
models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", "../results/yeast_smf_30_fs/mn", type="mn", num_processes=20)

# # run the null model
# models.cv.main("models.null_model", "cfgs/models/yeast_smf_full_model.json", "../results/yeast_smf_30_fs/null", num_processes=20)

# # analyze all models on training set
# analysis.tbl_model_comp.main("../results/yeast_smf_30_fs", 
#     "../tmp/yeast_smf_model_comp.xlsx", analysis.tbl_model_comp.SMF_LABELS)

# test
# models.cv.main("models.smf_nn", "cfgs/models/yeast_smf_full_model.json", 
#     "../results/task_yeast_smf_30_test/full", 
#     num_processes = 1,
#     target_col = "bin",
#     n_runs=1,
#     task_path="../generated-data/task_yeast_smf_30_train_test",
#     splits_path="../generated-data/splits/task_yeast_smf_30_train_test.npz")

# models.cv.main("models.smf_nn", "cfgs/models/yeast_smf_refined_model.json", 
#     "../results/task_yeast_smf_30_test/refined", 
#     num_processes = 1,
#     target_col = "bin",
#     n_runs=1,
#     task_path="../generated-data/task_yeast_smf_30_train_test",
#     splits_path="../generated-data/splits/task_yeast_smf_30_train_test.npz")

# models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", 
#     "../results/task_yeast_smf_30_test/orm", 
#     num_processes = 1,
#     target_col = "bin",
#     n_runs=1,
#     task_path="../generated-data/task_yeast_smf_30_train_test",
#     splits_path="../generated-data/splits/task_yeast_smf_30_train_test.npz")

models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", 
    "../results/task_yeast_smf_30_test/mn", 
    num_processes = 1,
    type="mn",
    target_col = "bin",
    n_runs=1,
    task_path="../generated-data/task_yeast_smf_30_train_test",
    splits_path="../generated-data/splits/task_yeast_smf_30_train_test.npz")


# models.cv.main("models.null_model", "cfgs/models/yeast_smf_refined_model.json", 
#     "../results/task_yeast_smf_30_test/null", 
#     num_processes = 1,
#     target_col = "bin",
#     n_runs=1,
#     task_path="../generated-data/task_yeast_smf_30_train_test",
#     splits_path="../generated-data/splits/task_yeast_smf_30_train_test.npz")


# # pombe
models.cv.main("models.null_model", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf/null")
models.cv.main("models.smf_nn", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf/refined")
models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", "../results/task_pombe_smf/orm")
models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", "../results/task_pombe_smf/mn", type="mn", num_processes=20)

# # human
models.cv.main("models.null_model", "cfgs/models/human_smf_refined_model.json", "../results/task_human_smf/null")
models.cv.main("models.smf_nn", "cfgs/models/human_smf_refined_model.json", "../results/task_human_smf/refined")
models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", "../results/task_human_smf/orm")
models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", "../results/task_human_smf/mn", type="mn", num_processes=20)

# # dro
models.cv.main("models.null_model", "cfgs/models/dro_smf_refined_model.json", "../results/task_dro_smf/null")
models.cv.main("models.smf_nn", "cfgs/models/dro_smf_refined_model.json", "../results/task_dro_smf/null_scrambled", scramble=True)
models.cv.main("models.smf_nn", "cfgs/models/dro_smf_refined_model.json", "../results/task_dro_smf/refined")
models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", "../results/task_dro_smf/orm")
models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", "../results/task_dro_smf/mn", type="mn", num_processes=20)

# # dro organsim viability
# models.cv.main("models.null_model", "cfgs/models/dro_smf_org_refined_model.json", "../results/task_dro_smf_org/null")
# models.cv.main("models.smf_nn", "cfgs/models/dro_smf_org_refined_model.json", "../results/task_dro_smf_org/null_scrambled", scramble=True)
# models.cv.main("models.smf_nn", "cfgs/models/dro_smf_org_refined_model.json", "../results/task_dro_smf_org/refined")
# models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_org_orm.json", "../results/task_dro_smf_org/orm")

# # human organsim viability
# models.cv.main("models.null_model", "cfgs/models/human_smf_org_refined_model.json", "../results/task_human_smf_org/null")
# models.cv.main("models.smf_nn", "cfgs/models/human_smf_org_refined_model.json", "../results/task_human_smf_org/null_scrambled", scramble=True)
# models.cv.main("models.smf_nn", "cfgs/models/human_smf_org_refined_model.json", "../results/task_human_smf_org/refined")
# models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_org_orm.json", "../results/task_human_smf_org/orm")

# # binary classification

# Yeast
models.cv.main("models.smf_nn", "cfgs/models/yeast_smf_refined_model.json", "../results/task_yeast_smf_30_binary/refined", 
    task_path="../generated-data/task_yeast_smf_30",
    splits_path="../generated-data/splits/task_yeast_smf_30_full.npz",
    target_col="is_viable",
    num_processes=20
)
models.cv.main("models.smf_nn", "cfgs/models/yeast_smf_full_model.json", "../results/task_yeast_smf_30_binary/full", 
    task_path="../generated-data/task_yeast_smf_30",
    splits_path="../generated-data/splits/task_yeast_smf_30_full.npz",
    target_col="is_viable",
    num_processes=20
)
models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", "../results/task_yeast_smf_30_binary/mn",
    task_path="../generated-data/task_yeast_smf_30",
    splits_path="../generated-data/splits/task_yeast_smf_30_full.npz",
    target_col="is_viable",
    type="mn",
    num_processes=20)
models.cv.main("models.null_model", "cfgs/models/yeast_smf_full_model.json", "../results/task_yeast_smf_30_binary/null",
    task_path="../generated-data/task_yeast_smf_30",
    splits_path="../generated-data/splits/task_yeast_smf_30_full.npz",
    target_col="is_viable",
    num_processes=20)

# # Pombe
# models.cv.main("models.null_model", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf_binary/null", 
#     targets_path = "../generated-data/targets/task_pombe_smf_bin_lethal.npz"
# )
# models.cv.main("models.smf_nn", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf_binary/null_scrambled", scramble=True, 
#     targets_path = "../generated-data/targets/task_pombe_smf_bin_lethal.npz")
# models.cv.main("models.smf_nn", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf_binary/refined", 
#     targets_path = "../generated-data/targets/task_pombe_smf_bin_lethal.npz")
models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", "../results/task_pombe_smf_binary/mn", 
    target_col="is_viable", type="mn", num_processes=20)


# # human
# models.cv.main("models.null_model", "cfgs/models/human_smf_refined_model.json", "../results/task_human_smf_binary/null", 
#     targets_path = "../generated-data/targets/task_human_smf_bin_lethal.npz"
# )
# models.cv.main("models.smf_nn", "cfgs/models/human_smf_refined_model.json", "../results/task_human_smf_binary/null_scrambled", scramble=True, 
#     targets_path = "../generated-data/targets/task_human_smf_bin_lethal.npz")
# models.cv.main("models.smf_nn", "cfgs/models/human_smf_refined_model.json", "../results/task_human_smf_binary/refined", 
#     targets_path = "../generated-data/targets/task_human_smf_bin_lethal.npz")
models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", "../results/task_human_smf_binary/mn", 
    target_col="is_viable", type="mn", num_processes=20)

# # dro
# models.cv.main("models.null_model", "cfgs/models/dro_smf_refined_model.json", "../results/task_dro_smf_binary/null", 
#     targets_path = "../generated-data/targets/task_dro_smf_bin_lethal.npz"
# )
# models.cv.main("models.smf_nn", "cfgs/models/dro_smf_refined_model.json", "../results/task_dro_smf_binary/null_scrambled", scramble=True, 
#     targets_path = "../generated-data/targets/task_dro_smf_bin_lethal.npz")
# models.cv.main("models.smf_nn", "cfgs/models/dro_smf_refined_model.json", "../results/task_dro_smf_binary/refined", 
#     targets_path = "../generated-data/targets/task_dro_smf_bin_lethal.npz")
models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", "../results/task_dro_smf_binary/mn", 
    target_col="is_viable", type="mn", num_processes=20)

#
# Human CA MA V
#

# # cell vs org lethal vs viable
# models.cv.main("models.smf_nn", "cfgs/models/human_smf_ca_ma_v_nn.json", 
#     "../results/task_human_ca_ma_v/refined", num_processes=20)
# models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_ca_ma_v_mn.json", 
#     "../results/task_human_ca_ma_v/mn", num_processes=20)
# models.cv.main("models.null_model", "cfgs/models/human_smf_ca_ma_v_nn.json", 
#     "../results/task_human_ca_ma_v/null", num_processes=20)

# # org lethal vs viable
# models.cv.main("models.smf_nn", "cfgs/models/human_smf_ca_ma_v_nn.json", 
#     "../results/task_human_ma_v/refined", num_processes=20,
#     task_path="../generated-data/task_human_smf_ma_v2",
#     splits_path="../generated-data/splits/task_human_smf_ma_v2_10reps_5folds_0.20valid.npz")
# models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_ca_ma_v_mn.json", 
#     "../results/task_human_ma_v/mn", num_processes=20,
#     task_path="../generated-data/task_human_smf_ma_v2",
#     splits_path="../generated-data/splits/task_human_smf_ma_v2_10reps_5folds_0.20valid.npz")
# models.cv.main("models.null_model", "cfgs/models/human_smf_ca_ma_v_nn.json", 
#     "../results/task_human_ma_v/null", num_processes=20,
#     task_path="../generated-data/task_human_smf_ma_v2",
#     splits_path="../generated-data/splits/task_human_smf_ma_v2_10reps_5folds_0.20valid.npz")


# #
# # Dro CA MA V
# #

# # cell vs org lethal vs viable
# models.cv.main("models.smf_nn", "cfgs/models/dro_smf_ca_ma_v_nn.json", 
#     "../results/task_dro_ca_ma_v/refined", num_processes=20)
# models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_ca_ma_v_mn.json", 
#     "../results/task_dro_ca_ma_v/mn", num_processes=20, type="orm")
# models.cv.main("models.null_model", "cfgs/models/dro_smf_ca_ma_v_nn.json", 
#     "../results/task_dro_ca_ma_v/null", num_processes=20)


# # org lethal vs viable
# models.cv.main("models.smf_nn", "cfgs/models/dro_smf_ca_ma_v_nn.json", 
#     "../results/task_dro_ma_v/refined", num_processes=20,
#     task_path="../generated-data/task_dro_smf_ma_v",
#     splits_path="../generated-data/splits/task_dro_smf_ma_v_10reps_5folds_0.20valid.npz")
# models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_ca_ma_v_mn.json", 
#     "../results/task_dro_ma_v/mn", num_processes=20,
#     task_path="../generated-data/task_dro_smf_ma_v",
#     splits_path="../generated-data/splits/task_dro_smf_ma_v_10reps_5folds_0.20valid.npz")
# models.cv.main("models.null_model", "cfgs/models/dro_smf_ca_ma_v_nn.json", 
#     "../results/task_dro_ma_v/null", num_processes=20,
#     task_path="../generated-data/task_dro_smf_ma_v",
#     splits_path="../generated-data/splits/task_dro_smf_ma_v_10reps_5folds_0.20valid.npz")

