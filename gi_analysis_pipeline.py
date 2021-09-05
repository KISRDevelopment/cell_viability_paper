import os 
import subprocess 
import sys 
import json 
import shlex

import analysis.tbl_model_comp
import analysis.fig_cv_performance
import analysis.fig_gi_smf_matrix 
import analysis.fig_pairwise_feature_heatmap
import analysis.fig_spl
from utils.merge_excels import merge_excels
import analysis.fig_class_distrib

GI_LABELS = ['Negative', 'Neutral', 'Positive', 'Supp']
GI_LABELS_BINARY = ['Interacting', 'Neutral']

#analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid", "../figures/yeast_gi_model_comp.xlsx", GI_LABELS)
#analysis.tbl_model_comp.main("../results/task_yeast_gi_hybrid_binary", "../figures/yeast_gi_binary_model_comp.xlsx", GI_LABELS_BINARY)
#analysis.tbl_model_comp.main("../results/task_yeast_gi_costanzo", "../figures/yeast_gi_costanzo_model_comp.xlsx", GI_LABELS)
#analysis.tbl_model_comp.main("../results/task_pombe_gi", "../figures/pombe_gi_model_comp.xlsx", GI_LABELS)
#analysis.tbl_model_comp.main("../results/task_pombe_gi_binary", "../figures/pombe_gi_binary_model_comp.xlsx", GI_LABELS_BINARY)
#analysis.tbl_model_comp.main("../results/task_human_gi", "../figures/human_gi_binary_model_comp.xlsx", GI_LABELS_BINARY)
#analysis.tbl_model_comp.main("../results/task_dro_gi", "../figures/dro_gi_binary_model_comp.xlsx", GI_LABELS_BINARY)

# merge_excels([
#     ("../figures/yeast_gi_model_comp.xlsx", "S. cerevisiae Hybrid"),
#     ("../figures/yeast_gi_binary_model_comp.xlsx", "S. cerevisiae Hybrid (Binary)"),
#     ("../figures/yeast_gi_costanzo_model_comp.xlsx", "S. cerevisiae Costanzo"),
#     ("../figures/pombe_gi_model_comp.xlsx", "S. pombe"),
#     ("../figures/pombe_gi_binary_model_comp.xlsx", "S. pombe (Binary)"),
#     ("../figures/human_gi_binary_model_comp.xlsx", "H. sapiens (Binary)"),
#     ("../figures/dro_gi_binary_model_comp.xlsx", "D. melanogaster (Binary)"),
# ], '../figures/gi_results.xlsx', writing_kw_args={"index":False})

# analysis.tbl_model_comp.main("../results/gi_generalization/pombe*", "../figures/pombe_gi_generalization_model_comp.xlsx", GI_LABELS_BINARY, use_glob_spec=True)
# analysis.tbl_model_comp.main("../results/gi_generalization/human*", "../figures/human_gi_generalization_model_comp.xlsx", GI_LABELS_BINARY, use_glob_spec=True)
# analysis.tbl_model_comp.main("../results/gi_generalization/dro*", "../figures/dro_gi_generalization_model_comp.xlsx", GI_LABELS_BINARY, use_glob_spec=True)

# merge_excels([
#     ("../figures/pombe_gi_generalization_model_comp.xlsx", "S. pombe (Binary)"),
#     ("../figures/human_gi_generalization_model_comp.xlsx", "H. sapiens (Binary)"),
#     ("../figures/dro_gi_generalization_model_comp.xlsx", "D. melanogaster (Binary)")
# ], '../figures/gi_generalization_results.xlsx', writing_kw_args={"index":False})


# SMF Matrices
# analysis.fig_gi_smf_matrix.main("../generated-data/task_yeast_gi_hybrid", "../generated-data/task_yeast_smf_30", "../figures/yeast_gi_hybrid", False)
# analysis.fig_gi_smf_matrix.main("../generated-data/task_yeast_gi_hybrid", "../generated-data/task_yeast_smf_30", "../figures/yeast_gi_hybrid", True)
# analysis.fig_gi_smf_matrix.main("../generated-data/task_pombe_gi", "../generated-data/task_pombe_smf", "../figures/pombe_gi", True)
# analysis.fig_gi_smf_matrix.main("../generated-data/task_human_gi", "../generated-data/task_human_smf", "../figures/human_gi", True)
# analysis.fig_gi_smf_matrix.main("../generated-data/task_dro_gi", "../generated-data/task_dro_smf", "../figures/dro_gi", True)

# SPL
# analysis.fig_spl.main("../generated-data/task_yeast_gi_hybrid", "../generated-data/pairwise_features/ppc_yeast_shortest_path_len.npy", "%", "../figures/yeast_hybrid_gi_spl")

# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_gi_hybrid.json")
#analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_gi_costanzo.json")
#analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_gi_costanzo_binary.json")

#analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_gi_hybrid_binary_no_sgo.json")
#analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_gi_hybrid_binary_no_sgo_test.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/pombe_gi.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/pombe_gi_binary.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_gi.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_gi.json")

# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/pombe_gi_binary_no_sgo.json")
#analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_gi_no_sgo.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_gi_no_sgo.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_gi_pombe.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_gi_human.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_gi_dro.json")


# Analyze pairwise LID
# Analyze SPL 
# Analyze SMF
# Analyze Class Distribution

# if not os.path.exists('../figures/analysis_yeast_hybrid_gi'):
#     os.makedirs('../figures/analysis_yeast_hybrid_gi')

#analysis.fig_pairwise_feature_heatmap.main("../generated-data/task_yeast_gi_hybrid", "../generated-data/features/ppc_yeast_topology.npz", "lid", "LID", "../figures/analysis_yeast_hybrid_gi/lid.png")
# analysis.fig_gi_smf_matrix.main("../generated-data/task_yeast_gi_hybrid", "../generated-data/task_yeast_smf_30", "../figures/analysis_yeast_hybrid_gi/smf", False)
# analysis.fig_spl.main("../generated-data/task_yeast_gi_hybrid", "../generated-data/pairwise_features/ppc_yeast_shortest_path_len.npy", "%", "../figures/analysis_yeast_hybrid_gi/spl")


# if not os.path.exists('../figures/analysis_pombe_gi'):
#     os.makedirs('../figures/analysis_pombe_gi')

#analysis.fig_pairwise_feature_heatmap.main("../generated-data/task_pombe_gi", "../generated-data/features/ppc_pombe_topology.npz", "lid", "LID", "../figures/analysis_pombe_gi/lid.png")
# analysis.fig_gi_smf_matrix.main("../generated-data/task_pombe_gi", "../generated-data/task_pombe_smf", "../figures/analysis_pombe_gi/smf", False)
# analysis.fig_spl.main("../generated-data/task_pombe_gi", "../generated-data/pairwise_features/ppc_pombe_shortest_path_len.npy", "%", "../figures/analysis_pombe_gi/spl")

# analysis.fig_class_distrib.main('cfgs/fig_class_distrib/class_distrib_gi.json')
# analysis.fig_class_distrib.main('cfgs/fig_class_distrib/class_distrib_gi_nonbinary.json')