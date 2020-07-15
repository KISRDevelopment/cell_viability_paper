import os 
import subprocess 
import sys 
import json 
import shlex

import utils.map_go_ids_to_names
import ppc_creation.ppc
import feature_preprocessing.topology
import feature_preprocessing.sgo 
import feature_preprocessing.redundancy
import tasks.pombe_smf

import utils.bin_simple
import utils.cv_simple
import models.cv 

gpath = "../generated-data/ppc_pombe"
pombe_smf_task = "../generated-data/task_pombe_smf"

if not os.path.exists('../generated-data/features'):
    os.makedirs('../generated-data/features')
if not os.path.exists('../generated-data/splits'):
    os.makedirs('../generated-data/splits')
if not os.path.exists('../generated-data/targets'):
    os.makedirs('../generated-data/targets')

# 1. Create mapping between GOIDs and Names
utils.map_go_ids_to_names.main()

# 2. Create pombe PPC network
ppc_creation.ppc.main("pombe", gpath)

# 3. Create Features

subprocess.call(shlex.split("../tools/owltools ../tools/go.obo --gaf ../data-sources/pombe/pombase.gaf --map2slim --subset goslim_generic --write-gaf ../tmp/pombase.sgo.gaf"))
feature_preprocessing.sgo.main(gpath, "../tmp/pombase.sgo.gaf", 1)
feature_preprocessing.topology.main(gpath, ['lid'])
feature_preprocessing.redundancy.main("pombe", gpath)

# 4. Create Task
tasks.pombe_smf.main(gpath)

# 5. Create Targets & CV Splits
utils.bin_simple.main(pombe_smf_task)
utils.cv_simple.main(pombe_smf_task, 10, 5, 0.2)

# # 8. Execute CV on refined, or, null models
models.cv.main("models.null_model", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf/null")
models.cv.main("models.smf_nn", "cfgs/models/pombe_smf_refined_model_scrambled.json", "../results/task_pombe_smf/null_scarmbled")
models.cv.main("models.smf_nn", "cfgs/models/pombe_smf_refined_model.json", "../results/task_pombe_smf/refined")
models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", "../results/task_pombe_smf/orm")
