import os 
import subprocess 
import sys 
import json 
import shlex
import glob 
import os 
import numpy as np 
import models.cv 
import scipy.stats as stats
import analysis.fig_interpretation

def load_cfg(path):
    with open(path, 'r') as f:
        return json.load(f)
# SMF Interpretation
models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", 
    "../results/smf_interpretation/yeast_orm", interpreation=True, num_processes=10, epochs=500)

models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", 
    "../results/smf_interpretation/pombe_orm", interpreation=True, num_processes=10, epochs=500)

models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", 
    "../results/smf_interpretation/human_orm", interpreation=True, num_processes=10, epochs=500)

models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", 
    "../results/smf_interpretation/dro_orm", interpreation=True, num_processes=10, epochs=500)

analysis.fig_interpretation.main(load_cfg('cfgs/fig_interpretation/smf.json'))


# SMF Organismal Cell Lethal Interpreation
models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_cell_org_lethal_orm.json", 
    "../results/smf_org_interpretation/human", interpreation=True, num_processes=20, epochs=150)

models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_cell_org_lethal_orm.json", 
    "../results/smf_org_interpretation/dro", interpreation=True, num_processes=20, epochs=150)
analysis.fig_interpretation.main(load_cfg('cfgs/fig_interpretation/smf_cell_org_lethal.json'))


# GI Binary Interpreation
models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/gi_interpretation/yeast_mn", interpreation=True, 
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz",
    num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
    "../results/gi_interpretation/pombe_mn", interpreation=True, 
    targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz",
    num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/human_gi_mn.json", 
    "../results/gi_interpretation/human_mn", interpreation=True, num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/dro_gi_mn.json", 
    "../results/gi_interpretation/dro_mn", interpreation=True, num_processes=20, epochs=50)

analysis.fig_interpretation.main(load_cfg('cfgs/fig_interpretation/gi_binary.json'))
