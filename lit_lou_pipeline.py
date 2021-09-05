import os 
import models.cv 
import feature_preprocessing.yeast_idc


# compute idc feature
feature_preprocessing.yeast_idc.main("../generated-data/ppc_yeast")

# execute CV
models.cv.main("models.smf_ordinal", 
    "cfgs/models/yeast_lid_idc.json", 
    "../results/lit_lou2015/lid_idc", 
    task_path="../generated-data/task_yeast_smf_30",
    splits_path="../generated-data/splits/task_yeast_smf_30_full.npz",
    target_col="is_viable",
    num_processes=20)

