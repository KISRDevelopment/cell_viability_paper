import numpy as np 
import pandas as pd 
import lit_tasks.campos
import os 
import utils.cv_simple
import models.cv 
import feature_preprocessing.amino_acid_features

# if not os.path.exists('../generated-data/lit_tasks'):
#     os.makedirs('../generated-data/lit_tasks')
# if not os.path.exists('../generated-data/lit_splits'):
#     os.makedirs('../generated-data/lit_splits')

# # Generate Dataset
# lit_tasks.campos.main("../generated-data/lit_tasks/campos2019")

# # Split it in 80% and 20% proportions
# utils.cv_simple.main("../generated-data/lit_tasks/campos2019", 1, 5, 0.2, "../generated-data/lit_splits/campos2019")

# # Execute models
# models.cv.main("models.smf_nn", 
#     "cfgs/models/yeast_smf_refined_model.json", 
#     "../results/lit_campos2019/refined",
#     num_processes=20,
#     task_path="../generated-data/lit_tasks/campos2019",
#     splits_path="../generated-data/lit_splits/campos2019.npz")

# models.cv.main("models.smf_ordinal", 
#     "cfgs/models/yeast_smf_orm.json", 
#     "../results/lit_campos2019/orm", 
#     task_path="../generated-data/lit_tasks/campos2019",
#     splits_path="../generated-data/lit_splits/campos2019.npz",
#     num_processes=20)


#feature_preprocessing.amino_acid_features.main("../generated-data/ppc_yeast", "../tmp/amino_acid_features.Rda")
models.cv.main("models.smf_nn", 
    "cfgs/models/yeast_campos.json", 
    "../results/lit_campos2019/campos",
    num_processes=20)
