import os 
import subprocess 
import sys 
import json 
import shlex
import numpy as np
import models.cv 
import analysis.fig_cv_performance
import analysis.fig_pairwise_feature_violin
import analysis.fig_tgi_smf_matrix_alt
import analysis.tbl_model_comp
import analysis.fig_interpretation
import utils.make_cfgs
import utils.make_gi_model_combs
import utils.make_smf_single_feature_sweeps
import models.multiple_cv

def load_cfg(path):
    with open(path, 'r') as f:
        return json.load(f)

#
# Feature Selection CV
#

# utils.make_gi_model_combs.feature_groups['triplet'] = ['triplet_const']
# base_cfg = load_cfg("cfgs/models/yeast_tgi_full_model.json")
# utils.make_gi_model_combs.main(base_cfg, "../tmp/model_cfgs/yeast_tgi")
# models.multiple_cv.main("models.tgi_nn", "../tmp/model_cfgs/yeast_tgi", 
#    "../results/task_yeast_tgi_feature_selection", 5, 
#    n_runs=40, exclude=lambda s: 'triplet' not in s)

# utils.make_cfgs.main("cfgs/models/sweep_yeast_tgi_pairwise_cfgs.json", "../tmp/model_cfgs/yeast_tgi_pairwisesweep")
# models.multiple_cv.main("models.tgi_nn", "../tmp/model_cfgs/yeast_tgi_pairwisesweep", 
#     "../results/task_yeast_tgi_feature_selection", 10, n_runs=40)


# utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_tgi/topology~go~pairwise~smf~triplet.json", "topology", "../tmp/model_cfgs/tgi_topology_sweep")
# models.multiple_cv.main("models.tgi_nn", "../tmp/model_cfgs/tgi_topology_sweep", 
#     "../results/task_yeast_tgi_feature_selection", 10, n_runs=40)

# move all feature selection results to task_yeast_tgi
# then do this
# analysis.tbl_model_comp.main("../results/task_yeast_tgi", 
#     "../figures/task_yeast_tgi_feature_selection.xlsx", ['Interacting', 'Neutral'])


#
# Cross validation performance
#

"""
models.cv.main("models.tgi_nn", "cfgs/models/yeast_tgi_full_model.json", 
    "../results/task_yeast_tgi/full", 
    num_processes=5)

models.cv.main("models.tgi_nn", "cfgs/models/yeast_tgi_refined_model.json", 
    "../results/task_yeast_tgi/refined", 
    num_processes=20)

models.cv.main("models.tgi_mn", "cfgs/models/yeast_tgi_mn.json", 
    "../results/task_yeast_tgi/mn", 
    num_processes=20)

models.cv.main("models.tgi_nn", "cfgs/models/yeast_tgi_refined_model.json", 
    "../results/task_yeast_tgi/null", 
    num_processes=20, scramble=True)

#
# Triple GI Interpretation
#

models.cv.main("models.tgi_mn", 
    "cfgs/models/yeast_tgi_mn.json", 
    "../results/tgi_interpretation/yeast_tgi_mn", 
    interpreation=True,
    num_processes=20, epochs=50)
analysis.fig_interpretation.main(load_cfg("cfgs/fig_interpretation/tgi.json"))

#
# Figures
#

analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_tgi.json")
"""

analysis.fig_pairwise_feature_violin.main("../generated-data/task_yeast_tgi", 
    "../generated-data/features/ppc_yeast_topology.npz", 
    "Sum LID", "../figures/yeast_tgi_sum_lid.png", 
    np.array(['Negative', 'Neutral']),
    colors=['magenta', 'cyan'],
    star_colors=['magenta', '#007bff'])
analysis.fig_tgi_smf_matrix_alt.main(
    "../generated-data/task_yeast_tgi",
    "../generated-data/task_yeast_smf_30",
    "../figures/yeast_tgi_smf"
)

analysis.fig_pairwise_feature_violin.main("../generated-data/task_ppc_yeast_pseudo_triplets", 
    "../generated-data/features/ppc_yeast_topology.npz", 
    "Sum LID", "../figures/yeast_pseudo_triplets_sum_lid.png", 
    np.array(['Within', 'Across']),
    colors=['magenta', 'cyan'],
    star_colors=['magenta', '#007bff'])

analysis.fig_tgi_smf_matrix_alt.BINARY_BIN_LABELS = ['Within', 'Across']
analysis.fig_tgi_smf_matrix_alt.main(
    "../generated-data/task_ppc_yeast_pseudo_triplets",
    "../generated-data/task_yeast_smf_30",
    "../figures/yeast_pseudo_triplets_smf"
)