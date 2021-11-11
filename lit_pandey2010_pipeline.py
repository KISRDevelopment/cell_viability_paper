import numpy as np 
import pandas as pd 
import os 
import models.cv 
import feature_preprocessing.pairwise_common_functions
import feature_preprocessing.pairwise_go_semsim
import feature_preprocessing.pairwise_pathway_comembership
import feature_preprocessing.pairwise_overlay

# generate the pairedwise shared GO terms feature
# feature_preprocessing.pairwise_common_functions.main()
# feature_preprocessing.pairwise_go_semsim.main("biological_process")
# feature_preprocessing.pairwise_go_semsim.main("cellular_component")
# feature_preprocessing.pairwise_pathway_comembership.main()
# feature_preprocessing.pairwise_overlay.main("../generated-data/ppc_yeast", "../generated-data/pairwise_features/ppc_yeast_semsim_biological_process.npy", 
#     "../generated-data/pairwise_features/ppc_yeast_overlay_ppi_semsim_biological_process.npy"
# )

models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_pandey2010.json", 
    "../results/lit_pandey2010/mnmc.slif", 
    num_processes=10,
    task_path="../generated-data/task_yeast_gi_hybrid",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
    target_col = "is_not_negative")

