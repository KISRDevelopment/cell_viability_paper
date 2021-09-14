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
smf_cfg_binary = load_cfg('cfgs/fig_interpretation/smf_binary.json')
smf_org_cfg = load_cfg('cfgs/fig_interpretation/smf_ca_ma_v.json')
gi_cfg = load_cfg('cfgs/fig_interpretation/gi_binary.json')
gi_cfg_yeast = load_cfg('cfgs/fig_interpretation/gi_binary_yeast_only.json')
gi_cfg_nonbinary = load_cfg('cfgs/fig_interpretation/gi_nonbinary.json')
gi_cfg_no_sgo = load_cfg('cfgs/fig_interpretation/gi_binary_no_sgo.json')
smf_cfg_mn = load_cfg("cfgs/fig_interpretation/smf_mn.json")

# #SMF Interpretation
# models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", 
#     "../results/smf_interpretation/yeast_orm", interpreation=True, num_processes=10, epochs=500)

# models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", 
#     "../results/smf_interpretation/pombe_orm", interpreation=True, num_processes=10, epochs=500)

# models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", 
#     "../results/smf_interpretation/human_orm", interpreation=True, num_processes=10, epochs=500)

# models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", 
#     "../results/smf_interpretation/dro_orm", interpreation=True, num_processes=10, epochs=500)
#analysis.fig_interpretation.main(smf_cfg)

# SMF Interpration MN
# models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", 
#     "../results/smf_interpretation/yeast_mn", interpreation=True, num_processes=10, epochs=500, type="mn", target_col="bin")

# models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", 
#     "../results/smf_interpretation/pombe_mn", interpreation=True, num_processes=10, epochs=500, type="mn", target_col="bin")

# models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", 
#     "../results/smf_interpretation/human_mn", interpreation=True, num_processes=10, epochs=500, type="mn", target_col="bin")

# models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", 
#     "../results/smf_interpretation/dro_mn", interpreation=True, num_processes=10, epochs=500, type="mn", target_col="bin")
#analysis.fig_interpretation.main(smf_cfg_mn)

# # SMF Binary Interpretation
# models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", 
#     "../results/smf_interpretation_binary/yeast_orm", 
#     targets_path="../generated-data/targets/task_yeast_smf_30_bin_lethal.npz",
#     interpreation=True, num_processes=10, epochs=500)

# models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", 
#     "../results/smf_interpretation_binary/pombe_orm", 
#     targets_path="../generated-data/targets/task_pombe_smf_bin_lethal.npz",
#     interpreation=True, num_processes=10, epochs=500)

# models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", 
#     "../results/smf_interpretation_binary/human_orm", 
#     targets_path="../generated-data/targets/task_human_smf_bin_lethal.npz",
#     interpreation=True, num_processes=10, epochs=500)

# models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", 
#     "../results/smf_interpretation_binary/dro_orm", 
#     targets_path="../generated-data/targets/task_dro_smf_bin_lethal.npz",
#     interpreation=True, num_processes=10, epochs=500)
analysis.fig_interpretation.main(smf_cfg_binary)


# # SMF Organismal Cell Lethal Interpreation
# models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_ca_ma_v_mn.json", 
#     "../results/smf_ca_ma_v_interpretation/human", interpreation=True, num_processes=20, epochs=150)

# models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_ca_ma_v_mn.json", 
#     "../results/smf_ca_ma_v_interpretation/dro", interpreation=True, num_processes=20, epochs=150)
analysis.fig_interpretation.main(smf_org_cfg)

# # merge everything into one file
# writer = pd.ExcelWriter('../figures/interpretation_smf.xlsx')
# for cfg, name in zip([smf_cfg, smf_cfg_binary, smf_org_cfg], ['SMF', 'SMF Binary', 'SMF Organismal']):
#     results_file = cfg['output_path'] + '.xlsx'
#     df = pd.read_excel(results_file, index_col=0, header=[0,1])
#     df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')

#     df.to_excel(writer, sheet_name=name, index=True)
# writer.save()

# GI non-binary interpretation
# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/gi_interpretation/yeast_mn_nonbinary", interpreation=True, 
#     targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_simple.npz",
#     num_processes=20, epochs=50)

# models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
#     "../results/gi_interpretation/pombe_mn_nonbinary", interpreation=True, 
#     targets_path="../generated-data/targets/task_pombe_gi_bin_simple.npz",
#     num_processes=20, epochs=50)
# analysis.fig_interpretation.main(gi_cfg_nonbinary)

# # Binary Interpreation

# costanzo_task_path = "../generated-data/task_yeast_gi_costanzo"
# costanzo_targets_path = "../generated-data/targets/task_yeast_gi_costanzo_bin_interacting.npz"
# costanzo_splits_path = "../generated-data/splits/task_yeast_gi_costanzo_10reps_4folds_0.20valid.npz"

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/gi_interpretation/yeast_costanzo_mn", 
#     interpreation=True, 
#     task_path = costanzo_task_path,
#     targets_path = costanzo_targets_path,
#     splits_path = costanzo_splits_path,
#     num_processes=20, epochs=50)

# models.cv.main("models.gi_mn", "cfgs/models/yeast_gi_mn.json", 
#     "../results/gi_interpretation/yeast_mn", interpreation=True, 
#     targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz",
#     num_processes=20, epochs=50)

# models.cv.main("models.gi_mn", "cfgs/models/pombe_gi_mn.json", 
#     "../results/gi_interpretation/pombe_mn", interpreation=True, 
#     targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz",
#     num_processes=20, epochs=50)

# models.cv.main("models.gi_mn", "cfgs/models/human_gi_mn.json", 
#     "../results/gi_interpretation/human_mn", interpreation=True, target_col="is_neutral", num_processes=20, epochs=50)

# models.cv.main("models.gi_mn", "cfgs/models/dro_gi_mn.json", 
#     "../results/gi_interpretation/dro_mn", interpreation=True, target_col="is_neutral", num_processes=20, epochs=50)
analysis.fig_interpretation.main(gi_cfg)

# # merge everything into one file
# writer = pd.ExcelWriter('../figures/interpretation_gi.xlsx')
# for cfg, name in zip([gi_cfg_nonbinary, gi_cfg], ['GI', 'GI Binary']):

#     results_file = cfg['output_path'] + '.xlsx'
#     df = pd.read_excel(results_file, index_col=0, header=[0,1])
#     df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
#     print(df.columns)
#     df.to_excel(writer, sheet_name=name, index=True)
# writer.save()