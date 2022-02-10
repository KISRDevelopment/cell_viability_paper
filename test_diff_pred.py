import utils.evaluate_cross_pred
import json 
import os 
import numpy as np
import models.cv



# create GO without some terms
exclude_GOIDS = ['GO:0016791', 'GO:0016301', 'GO:0008134']
d = np.load("../generated-data/features/ppc_yeast_common_sgo.npz")
labels = d['feature_labels']
ix = [i for i in range(d['F'].shape[1]) if labels[i] not in exclude_GOIDS]
F = d['F'][:,ix]
labels = labels[ix]
np.savez("../generated-data/features/ppc_yeast_common_sgo_strict.npz", F=F, feature_labels=labels)

models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/task_yeast_gi_hybrid_binary/mn_strict", 
    num_processes=20,
    target_col="is_neutral",
    task_path="../generated-data/task_yeast_gi_hybrid",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
    spec=[{
            "name" : "smf",
            "path" : "../generated-data/features/ppc_yeast_smf_binned.npz",
            "processor" : "BinnedSmfProcessor"
        },
        {
            "name" : "sgo",
            "path" : "../generated-data/features/ppc_yeast_common_sgo_strict.npz",
            "processor" : "GoProcessorSum"
        },
    ]
)


#utils.evaluate_cross_pred.main("../results/task_yeast_gi_hybrid_binary/mn", "../results/mn_crosspred.csv")
utils.evaluate_cross_pred.main("../results/task_yeast_gi_hybrid_binary/mn_strict", "../results/mn_strict_crosspred.csv")
