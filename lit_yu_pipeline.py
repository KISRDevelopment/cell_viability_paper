import numpy as np 
import pandas as pd 
import os 
import models.cv 
import feature_preprocessing.sgo

# generate the pairedwise shared sGO terms feature
# feature_preprocessing.sgo.main("../generated-data/ppc_yeast", "../data-sources/yeast/sgd.gaf", 
#     gene_name_col=0,
#     output_path="../generated-data/features/ppc_yeast_full_go",
#     annotations_reader=feature_preprocessing.sgo.read_annotations_yeast)

models.cv.main("models.gi_nn", "cfgs/models/yeast_gi_shared_full_go.json", 
    "../results/lit_yu2016/yu", 
    num_processes=1,
    task_path="../generated-data/task_yeast_gi_hybrid",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
    target_col = "is_not_negative")
