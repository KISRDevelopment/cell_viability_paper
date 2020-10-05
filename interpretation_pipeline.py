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
import pandas as pd 
def load_cfg(path):
    with open(path, 'r') as f:
        return json.load(f)

smf_cfg = load_cfg('cfgs/fig_interpretation/smf.json')
smf_org_cfg = load_cfg('cfgs/fig_interpretation/smf_cell_org_lethal.json')
gi_cfg = load_cfg('cfgs/fig_interpretation/gi_binary.json')
gi_cfg_no_sgo = load_cfg('cfgs/fig_interpretation/gi_binary_no_sgo.json')


# # SMF Interpretation
# models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", 
#     "../results/smf_interpretation/yeast_orm", interpreation=True, num_processes=10, epochs=500)

# models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", 
#     "../results/smf_interpretation/pombe_orm", interpreation=True, num_processes=10, epochs=500)

# models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", 
#     "../results/smf_interpretation/human_orm", interpreation=True, num_processes=10, epochs=500)

# models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", 
#     "../results/smf_interpretation/dro_orm", interpreation=True, num_processes=10, epochs=500)

analysis.fig_interpretation.main(smf_cfg)


# # SMF Organismal Cell Lethal Interpreation
# models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_cell_org_lethal_orm.json", 
#     "../results/smf_org_interpretation/human", interpreation=True, num_processes=20, epochs=150)

# models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_cell_org_lethal_orm.json", 
#     "../results/smf_org_interpretation/dro", interpreation=True, num_processes=20, epochs=150)
#analysis.fig_interpretation.main(smf_org_cfg)

# GI non-binary interpretation
models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/gi_interpretation/yeast_mn_nonbinary", interpreation=True, 
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_simple.npz",
    num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
    "../results/gi_interpretation/pombe_mn_nonbinary", interpreation=True, 
    targets_path="../generated-data/targets/task_pombe_gi_bin_simple.npz",
    num_processes=20, epochs=50)

# # GI Binary Interpreation
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

analysis.fig_interpretation.main(gi_cfg)

#
# no SGO
#
models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
    "../results/gi_interpretation_no_sgo/yeast_mn", interpreation=True, 
    remove_specs=["sgo"],
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz",
    num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
    "../results/gi_interpretation_no_sgo/pombe_mn", interpreation=True, 
    remove_specs=["sgo"],
    targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz",
    num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/human_gi_mn.json", 
    "../results/gi_interpretation_no_sgo/human_mn",     remove_specs=["sgo"], interpreation=True, num_processes=20, epochs=50)

models.cv.main("models.gi_mn", "cfgs/models/dro_gi_mn.json", 

    "../results/gi_interpretation_no_sgo/dro_mn",     remove_specs=["sgo"], interpreation=True, num_processes=20, epochs=50)
analysis.fig_interpretation.main(gi_cfg_no_sgo)

# merge everything into one file
# writer = pd.ExcelWriter('../figures/interpretation_results.xlsx')
# for cfg, name in zip([smf_cfg, smf_org_cfg, gi_cfg], ['SMF', 'SMF Organismal', 'GI']):

#     results_file = cfg['output_path'] + '.xlsx'
#     df = pd.read_excel(results_file, index_col=0, header=[0,1])
#     df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
#     print(df.columns)
#     df.to_excel(writer, sheet_name=name, index=True)
# writer.save()