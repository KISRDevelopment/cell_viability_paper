import numpy as np 
import pandas as pd 
import os 
import utils.cv_simple
import models.cv 
import utils.yeast_name_resolver
import feature_preprocessing.pairwise_mistry2017_rma
import feature_preprocessing.diffslc_ppc

res = utils.yeast_name_resolver.NameResolver()

if not os.path.exists('../generated-data/lit_tasks'):
    os.makedirs('../generated-data/lit_tasks')
if not os.path.exists('../generated-data/lit_splits'):
    os.makedirs('../generated-data/lit_splits')


# compute coexpression
#feature_preprocessing.pairwise_mistry2017_rma.main("../generated-data/ppc_yeast")

# compute diffslc feature
#feature_preprocessing.diffslc_ppc.main("../generated-data/ppc_yeast", " ../generated-data/pairwise_features/ppc_yeast_rma_dcor.npy")

# execute CV
models.cv.main("models.smf_ordinal", 
    "cfgs/models/yeast_diffslc.json", 
    "../results/lit_mistry2017/diffslc", 
    task_path="../generated-data/task_yeast_smf_30",
    splits_path="../generated-data/splits/task_yeast_smf_30_full.npz",
    target_col="is_viable",
    num_processes=20)

