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

if not os.path.exists('../results/yeast_smf_figures'):
    os.makedirs('../results/yeast_smf_figures')

# # 1. Plot CS vs STD
# analysis.fig_cs_vs_std.main("../generated-data/task_yeast_smf_30", "../results/yeast_smf_figures/cs_vs_std.png", show=False)

# # 2. Plot lid & pident violins
# analysis.fig_feature_violin.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast_topology.npz", 11, "../results/yeast_smf_figures/lid_violin.png")
# analysis.fig_feature_violin.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast_redundancy.npz", 0, "../results/yeast_smf_figures/pident_violin.png")

# # 3. Topology and redundancy correlations
# analysis.fig_feature_corr.main("../generated-data/features/ppc_yeast_topology.npz", "../results/yeast_smf_figures/topology_corr.png", False)
# analysis.fig_feature_corr.main("../generated-data/features/ppc_yeast_redundancy.npz", "../results/yeast_smf_figures/redundancy_corr.png", False)

# # 4. GO Table
# analysis.tbl_go_vs_smf.main("../generated-data/task_yeast_smf_30", "../generated-data/features/ppc_yeast-sgo.npz", "../results/yeast_smf_figures/sgo.xlsx")

# 5. CV Report
analysis.tbl_model_comp.main("../results/task_yeast_smf_30", "../results/yeast_smf_figures/model_comp.xlsx")
