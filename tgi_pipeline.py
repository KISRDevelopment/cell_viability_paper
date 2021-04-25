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

def load_cfg(path):
    with open(path, 'r') as f:
        return json.load(f)


#
# Cross validation performance
#

# models.cv.main("models.tgi_nn", "cfgs/models/yeast_tgi_full_model.json", 
#     "../results/task_yeast_tgi/full", 
#     num_processes=5)

# models.cv.main("models.tgi_nn", "cfgs/models/yeast_tgi_refined_model.json", 
#     "../results/task_yeast_tgi/refined", 
#     num_processes=20)

# models.cv.main("models.tgi_mn", "cfgs/models/yeast_tgi_mn.json", 
#     "../results/task_yeast_tgi/mn", 
#     num_processes=20)

# models.cv.main("models.tgi_nn", "cfgs/models/yeast_tgi_refined_model.json", 
#     "../results/task_yeast_tgi/null", 
#     num_processes=20, scramble=True)

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

# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_tgi.json")
# analysis.fig_pairwise_feature_violin.main("../generated-data/task_yeast_tgi", 
#     "../generated-data/features/ppc_yeast_topology.npz", 
#     "Sum LID", "../figures/yeast_tgi_sum_lid.png", 
#     np.array(['-', 'N']),
#     colors=['magenta', 'cyan'],
#     star_colors=['magenta', '#007bff'])
# analysis.fig_tgi_smf_matrix_alt.main(
#     "../generated-data/task_yeast_tgi",
#     "../generated-data/task_yeast_smf_30",
#     "../figures/yeast_tgi_smf"
# )