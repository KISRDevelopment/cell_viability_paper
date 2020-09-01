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

if not os.path.exists('../results/yeast_smf_figures'):
    os.makedirs('../results/yeast_smf_figures')

analysis.fig_cs_vs_std.main("../generated-data/task_yeast_smf_30", "../results/yeast_smf_figures/cs_vs_std.png", show=False)
analysis.fig_feature_violin.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast_topology.npz", 11, "LID Score", "../results/yeast_smf_figures/lid_violin.png")
analysis.fig_feature_violin.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast_redundancy.npz", 0, "Percent Identity Score", "../results/yeast_smf_figures/pident_violin.png")

analysis.fig_feature_corr.main("../generated-data/features/ppc_yeast_topology.npz", "../results/yeast_smf_figures/topology_corr.png", False)
analysis.fig_feature_corr.main("../generated-data/features/ppc_yeast_redundancy.npz", "../results/yeast_smf_figures/redundancy_corr.png", False)

analysis.tbl_go_vs_smf.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast_common_sgo.npz", "../results/yeast_smf_figures/sgo.xlsx")
analysis.fig_go_vs_smf.main("../results/yeast_smf_figures/sgo.xlsx", "../results/yeast_smf_figures/sgo_vs_smf.png")

analysis.tbl_model_comp.main("../results/task_yeast_smf_30", "../results/yeast_smf_figures/model_comp.xlsx")

analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_smf.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/pombe_smf.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_smf.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_smf.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_smf_cell_org_lethal.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_smf_cell_org_lethal.json")

analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_pombe.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_human.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_dro.json")

analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_smf_binary.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/pombe_smf_binary.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/human_smf_binary.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/dro_smf_binary.json")

analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_pombe_binary.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_human_binary.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_dro_binary.json")