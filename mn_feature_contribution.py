import os 
import subprocess 
import sys 
import json 
import shlex

import models.cv 

spec = [
        {
            "name" : "topology",
            "path" : "../generated-data/features/ppc_yeast_topology.npz",
            "selected_features" : ["lid"]
        },
        {
            "name" : "redundancy",
            "selected_features" : ["pident"],
            "path" : "../generated-data/features/ppc_yeast_redundancy.npz"
        },
        {
            "name" : "go",
            "path" : "../generated-data/features/ppc_yeast_common_sgo.npz",
            "selected_features" : None
        }
]

models.cv.main("models.smf_ordinal", 
    "cfgs/models/yeast_smf_orm.json", 
    "../results/yeast_smf_30_fs/mn_no_topology", 
    type="mn", 
    num_processes=20,
    spec=spec[1:])

models.cv.main("models.smf_ordinal", 
    "cfgs/models/yeast_smf_orm.json", 
    "../results/yeast_smf_30_fs/mn_no_redundancy", 
    type="mn", 
    num_processes=20,
    spec=[spec[0], spec[2]])

models.cv.main("models.smf_ordinal", 
    "cfgs/models/yeast_smf_orm.json", 
    "../results/yeast_smf_30_fs/mn_no_sgo", 
    type="mn", 
    num_processes=20,
    spec=spec[:-1])
