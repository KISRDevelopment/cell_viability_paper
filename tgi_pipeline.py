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
import utils.make_cfgs


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

#analysis.fig_cv_performance.main("cfgs/fig_cv_performance/yeast_tgi.json")
# analysis.fig_pairwise_feature_violin.main("../generated-data/task_yeast_tgi", 
#     "../generated-data/features/ppc_yeast_topology.npz", 
#     "Sum LID", "../figures/yeast_tgi_sum_lid.png", 
#     np.array(['-', 'N']),
#     colors=['magenta', 'cyan'],
#     star_colors=['magenta', '#007bff'])
analysis.fig_tgi_smf_matrix_alt.main(
    "../generated-data/task_yeast_tgi",
    "../generated-data/task_yeast_smf_30",
    "../figures/yeast_tgi_smf"
)