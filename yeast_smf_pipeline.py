import os 
import subprocess 
import sys 
import json 
import shlex

import utils.map_go_ids_to_names
import ppc_creation.yeast_ppc
import feature_preprocessing.topology
import feature_preprocessing.yeast_abundance
import feature_preprocessing.yeast_localization
import feature_preprocessing.yeast_phosphotase
import feature_preprocessing.yeast_redundancy
import feature_preprocessing.yeast_sgo
import feature_preprocessing.yeast_transcription
import tasks.yeast_smf
import utils.bin_simple
import utils.cv_simple
import utils.make_smf_single_feature_sweeps
import utils.make_smf_model_combs
import models.multiple_cv
import models.cv 

gpath = "../generated-data/ppc_yeast"
yeast_smf_task = "../generated-data/task_yeast_smf_30"

if not os.path.exists('../generated-data/features'):
    os.makedirs('../generated-data/features')
if not os.path.exists('../generated-data/splits'):
    os.makedirs('../generated-data/splits')
if not os.path.exists('../generated-data/targets'):
    os.makedirs('../generated-data/targets')

# # 1. Create mapping between GOIDs and Names
# utils.map_go_ids_to_names.main()

# # 2. Create yeast PPC network
# ppc_creation.yeast_ppc.main(gpath)

# # 3. Create Features

# feature_preprocessing.yeast_abundance.main(gpath)
# feature_preprocessing.yeast_localization.main(gpath)
# feature_preprocessing.yeast_phosphotase.main(gpath, "../data-sources/yeast/kinase.txt")
# feature_preprocessing.yeast_phosphotase.main(gpath, "../data-sources/yeast/phosphotase.txt")
# feature_preprocessing.yeast_sgo.main(gpath)
# feature_preprocessing.yeast_transcription.main(gpath)
# feature_preprocessing.topology.main(gpath)
# feature_preprocessing.yeast_redundancy.main(gpath)

# # 4. Create Task
# tasks.yeast_smf.main(gpath)

# # 5. Create Targets & CV Splits
# utils.bin_simple.main(yeast_smf_task)
# utils.cv_simple.main(yeast_smf_task, 10, 5, 0.2)

# # 6. Create Model Combinations
# utils.make_smf_model_combs.main()
# utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_smf/topology~go~redundancy.json", "redundancy", "../tmp/model_cfgs/yeast_smf")
# utils.make_smf_single_feature_sweeps.main("../tmp/model_cfgs/yeast_smf/topology~go~redundancy.json", "topology", "../tmp/model_cfgs/yeast_smf")

# 7. Execute CV on all model combinations
#models.multiple_cv.main("models.smf_nn", "../tmp/model_cfgs/yeast_smf", "../results/task_yeast_smf_30")

# 8. Execute CV on null models
#models.cv.main("models.null_model", "cfgs/models/yeast_smf_full_model.json", "../results/task_yeast_smf_30/null")
#models.cv.main("models.smf_nn", "cfgs/models/yeast_smf_full_model_scrambled.json", "../results/task_yeast_smf_30/null_scarmbled")

# 9. Execute CV on OR model
models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", "../results/task_yeast_smf_30/orm")
