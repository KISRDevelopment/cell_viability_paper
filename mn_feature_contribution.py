import os 
import subprocess 
import sys 
import json 
import shlex

import models.cv 

# models.cv.main("models.smf_ordinal", 
#     "cfgs/models/yeast_smf_orm.json", 
#     "../results/yeast_smf_30_fs/mn_no_topology", 
#     type="mn", 
#     num_processes=20,
#     remove_specs=["topology"])

# models.cv.main("models.smf_ordinal", 
#     "cfgs/models/yeast_smf_orm.json", 
#     "../results/yeast_smf_30_fs/mn_no_redundancy", 
#     type="mn", 
#     num_processes=20,
#     remove_specs=["redundancy"])

# models.cv.main("models.smf_ordinal", 
#     "cfgs/models/yeast_smf_orm.json", 
#     "../results/yeast_smf_30_fs/mn_no_sgo", 
#     type="mn", 
#     num_processes=20,
#     remove_specs=["go"])

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_fs/mn_no_smf", 
#     num_processes=10,
#     remove_specs=["smf"])

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_fs/mn_no_sgo", 
#     num_processes=10,
#     remove_specs=["sgo"])

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/task_yeast_gi_hybrid_fs/mn_no_topology", 
#     num_processes=10,
#     remove_specs=["spl", "topology"])

models.cv.main("models.tgi_mn", "cfgs/models/yeast_tgi_mn.json", 
    "../results/task_yeast_tgi_fs/mn_no_topology", 
    remove_specs=["spl","topology"],
    num_processes=20)

models.cv.main("models.tgi_mn", "cfgs/models/yeast_tgi_mn.json", 
    "../results/task_yeast_tgi_fs/mn_no_smf", 
    remove_specs=["smf"],
    num_processes=20)

models.cv.main("models.tgi_mn", "cfgs/models/yeast_tgi_mn.json", 
    "../results/task_yeast_tgi_fs/mn_no_sgo", 

    remove_specs=["sgo"],
    num_processes=20)
