import os 
import subprocess 
import sys 
import json 
import shlex

import analysis.fig_cs_vs_std
import analysis.fig_feature_violin
import analysis.fig_feature_corr
import analysis.tbl_go_vs_smf
import analysis.tbl_model_comp
import analysis.fig_cv_performance
import analysis.fig_go_vs_smf
from utils.merge_excels import merge_excels
import analysis.fig_class_distrib
if not os.path.exists('../figures/yeast_smf_analysis'):
    os.makedirs('../figures/yeast_smf_analysis')

#analysis.fig_class_distrib.main('cfgs/fig_class_distrib/class_distrib_smf.json')
analysis.fig_class_distrib.main('cfgs/fig_class_distrib/class_distrib_org.json')

# analysis.fig_cs_vs_std.main("../generated-data/task_yeast_smf_30", "../results/yeast_smf_figures/cs_vs_std.png", show=False)
# analysis.fig_feature_violin.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast_topology.npz", 11, "LID Score", "../results/yeast_smf_figures/lid_violin.png")
# analysis.fig_feature_violin.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast_redundancy.npz", 0, "Percent Identity Score", "../results/yeast_smf_figures/pident_violin.png")

# analysis.fig_feature_corr.main("../generated-data/features/ppc_yeast_topology.npz", "../results/yeast_smf_figures/topology_corr.png", False)
# analysis.fig_feature_corr.main("../generated-data/features/ppc_yeast_redundancy.npz", "../results/yeast_smf_figures/redundancy_corr.png", False)

# analysis.tbl_go_vs_smf.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast_common_sgo.npz", "../figures/yeast_smf_analysis/sgo.xlsx")
# analysis.fig_go_vs_smf.main("../figures/yeast_smf_analysis/sgo.xlsx", "../figures/yeast_smf_analysis/sgo_vs_smf.png")

# analysis.fig_feature_corr.main("../generated-data/features/ppc_yeast_topology.npz", "../results/yeast_smf_figures/topology_corr.png", False)
# analysis.fig_feature_corr.main("../generated-data/features/ppc_yeast_redundancy.npz", "../results/yeast_smf_figures/redundancy_corr.png", False)

# analysis.tbl_go_vs_smf.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast_common_sgo.npz", "../results/yeast_smf_figures/sgo.xlsx")
# analysis.fig_go_vs_smf.main("../results/yeast_smf_figures/sgo.xlsx", "../results/yeast_smf_figures/sgo_vs_smf.png")

# analysis.tbl_model_comp.main("../results/task_yeast_smf_30", "../figures/yeast_smf_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_yeast_smf_30_binary", "../figures/yeast_smf_binary_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_pombe_smf", "../figures/pombe_smf_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_pombe_smf_binary", "../figures/pombe_smf_binary_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_human_smf", "../figures/human_smf_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_human_smf_binary", "../figures/human_smf_binary_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_dro_smf", "../figures/dro_smf_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_dro_smf_binary", "../figures/dro_smf_binary_model_comp.xlsx")

# analysis.tbl_model_comp.main("../results/task_human_smf_cell_org_lethal", "../figures/human_smf_cell_org_lethal_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_human_smf_org", "../figures/human_smf_org_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_dro_smf_cell_org_lethal", "../figures/dro_smf_cell_org_lethal_model_comp.xlsx")
# analysis.tbl_model_comp.main("../results/task_dro_smf_org", "../figures/dro_smf_org_model_comp.xlsx")

# merge_excels([
#     ("../figures/yeast_smf_model_comp.xlsx", "S. cerevisiae"),
#     ("../figures/yeast_smf_binary_model_comp.xlsx", "S. cerevisiae (Binary)"),
#     ("../figures/pombe_smf_model_comp.xlsx", "S. pombe"),
#     ("../figures/pombe_smf_binary_model_comp.xlsx", "S. pombe (Binary)"),
#     ("../figures/human_smf_model_comp.xlsx", "H. sapiens"),
#     ("../figures/human_smf_binary_model_comp.xlsx", "H. sapiens (Binary)"),
#     ("../figures/dro_smf_model_comp.xlsx", "D. melanogaster"),
#     ("../figures/dro_smf_binary_model_comp.xlsx", "D. melanogaster (Binary)"),
#     ("../figures/human_smf_org_model_comp.xlsx", "H. sapiens (Organismal Lethal)"),
#     ("../figures/human_smf_cell_org_lethal_model_comp.xlsx", "H. sapiens (Cell and Organismal Lethal)"),
#     ("../figures/dro_smf_org_model_comp.xlsx", "D. melanogaster (Organismal Lethal)"),
#     ("../figures/dro_smf_cell_org_lethal_model_comp.xlsx", "D. melanogaster (Cell and Organismal Lethal)"),
# ], '../figures/smf_results.xlsx', writing_kw_args={"index":False})

# analysis.tbl_model_comp.main("../results/smf_generalization/pombe*", "../figures/pombe_smf_generalization_model_comp.xlsx", use_glob_spec=True)
# analysis.tbl_model_comp.main("../results/smf_generalization/human*", "../figures/human_smf_generalization_model_comp.xlsx", use_glob_spec=True)
# analysis.tbl_model_comp.main("../results/smf_generalization/dro*", "../figures/dro_smf_generalization_model_comp.xlsx", use_glob_spec=True)
# analysis.tbl_model_comp.main("../results/smf_generalization_binary/pombe*", "../figures/pombe_smf_generalization_binary_model_comp.xlsx", use_glob_spec=True)
# analysis.tbl_model_comp.main("../results/smf_generalization_binary/human*", "../figures/human_smf_generalization_binary_model_comp.xlsx", use_glob_spec=True)
# analysis.tbl_model_comp.main("../results/smf_generalization_binary/dro*", "../figures/dro_smf_generalization_binary_model_comp.xlsx", use_glob_spec=True)
# merge_excels([
#     ("../figures/pombe_smf_generalization_model_comp.xlsx", "S. pombe"),
#     ("../figures/pombe_smf_generalization_binary_model_comp.xlsx", "S. pombe (Binary)"),
#     ("../figures/human_smf_generalization_model_comp.xlsx", "H. sapiens"),
#     ("../figures/human_smf_generalization_binary_model_comp.xlsx", "H. sapiens (Binary)"),
#     ("../figures/dro_smf_generalization_model_comp.xlsx", "D. melanogaster"),
#     ("../figures/dro_smf_generalization_binary_model_comp.xlsx", "D. melanogaster (Binary)")
# ], '../figures/smf_generalization_results.xlsx', writing_kw_args={"index":False})


# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_smf.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/pombe_smf.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_smf.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_smf.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_smf_cell_org_lethal.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_smf_cell_org_lethal.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_smf_org.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_smf_org.json")

# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_pombe.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_human.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_dro.json")

# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_smf_binary.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/pombe_smf_binary.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_smf_binary.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_smf_binary.json")

# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_pombe_binary.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_human_binary.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_dro_binary.json")