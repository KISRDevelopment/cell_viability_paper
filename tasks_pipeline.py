import os 
import subprocess 
import sys 
import json 
import shlex

import tasks.yeast_smf
import tasks.pombe_smf 
import tasks.human_smf
import tasks.human_org
import tasks.dro_smf
import tasks.dro_org_smf
import tasks.pombe_gi
import tasks.yeast_gi_costanzo
import tasks.yeast_gi_hybrid
import tasks.biogrid_plus_negative_sampling
import utils.make_biogrid_dataset
import utils.make_costanzo_dataset
import utils.bin_simple
import utils.cv_simple
import utils.cv_gi
import utils.bin_interacting
import utils.make_fb_dataset
import ppc_creation.ppc
import utils.bin_lethal
import tasks.cell_org_lethal

if not os.path.exists('../generated-data/splits'):
    os.makedirs('../generated-data/splits')
if not os.path.exists('../generated-data/targets'):
    os.makedirs('../generated-data/targets')

# # PPC
# ppc_creation.ppc.main("yeast", "../generated-data/ppc_yeast")
# ppc_creation.ppc.main("pombe", "../generated-data/ppc_pombe")
# ppc_creation.ppc.main("human", "../generated-data/ppc_human")
# ppc_creation.ppc.main("dro", "../generated-data/ppc_dro")

# # smf tasks

# gpath = "../generated-data/ppc_yeast"
# smf_task_path = "../generated-data/task_yeast_smf_30"
# tasks.yeast_smf.main(gpath, 30, smf_task_path)
# utils.bin_simple.main(smf_task_path)
# utils.cv_simple.main(smf_task_path, 10, 5, 0.2)
# utils.bin_lethal.main(smf_task_path)

# gpath = "../generated-data/ppc_pombe"
# smf_task_path = "../generated-data/task_pombe_smf"
# tasks.pombe_smf.main(gpath, smf_task_path)
# utils.bin_simple.main(smf_task_path)
# utils.cv_simple.main(smf_task_path, 10, 5, 0.2)
# utils.bin_lethal.main(smf_task_path)

# gpath = "../generated-data/ppc_human"
# smf_task_path = "../generated-data/task_human_smf"
# tasks.human_smf.main(gpath, smf_task_path)
# utils.bin_simple.main(smf_task_path)
# utils.cv_simple.main(smf_task_path, 10, 5, 0.2)
# utils.bin_lethal.main(smf_task_path)

# gpath = "../generated-data/ppc_human"
# cell_smf_task_path = "../generated-data/task_human_smf"
# smf_task_path = "../tmp/task_human_smf_org"
# tasks.human_org.main(gpath, cell_smf_task_path, smf_task_path)
#utils.bin_simple.main(smf_task_path)
#utils.cv_simple.main(smf_task_path, 10, 5, 0.2)

# gpath = "../generated-data/ppc_dro"
# smf_task_path = "../generated-data/task_dro_smf"
# tasks.dro_smf.main(gpath, smf_task_path)
# utils.bin_simple.main(smf_task_path)
# utils.cv_simple.main(smf_task_path, 10, 5, 0.2)
# utils.bin_lethal.main(smf_task_path)

# gpath = "../generated-data/ppc_dro"
# smf_task_path = "../tmp/task_dro_smf_org"
# tasks.dro_org_smf.main(gpath, "../data-sources/dro/Essential genes.csv", smf_task_path)
# utils.bin_simple.main(smf_task_path)
# utils.cv_simple.main(smf_task_path, 10, 5, 0.2)

# smf_task_path = "../generated-data/task_human_cell_org_lethal"
# tasks.cell_org_lethal.main("../generated-data/ppc_human", "../generated-data/task_human_smf", 
#     "../generated-data/task_human_smf_org", smf_task_path)
# utils.bin_simple.main(smf_task_path)
# utils.cv_simple.main(smf_task_path, 10, 5, 0.2)


# smf_task_path = "../generated-data/task_dro_cell_org_lethal"
# tasks.cell_org_lethal.main("../generated-data/ppc_dro", "../generated-data/task_dro_smf", 
#     "../generated-data/task_dro_smf_org", smf_task_path)
# utils.bin_simple.main(smf_task_path)
# utils.cv_simple.main(smf_task_path, 10, 5, 0.2)

# # GI tasks

# utils.make_costanzo_dataset.main('../generated-data/costanzo_gi')
# utils.make_biogrid_dataset.main(559292, 3, '../generated-data/biogrid_yeast')
# utils.make_biogrid_dataset.main(284812, 2, '../generated-data/biogrid_pombe')
# utils.make_biogrid_dataset.main(9606, 1, '../generated-data/biogrid_human')
# utils.make_fb_dataset.main('../generated-data/fb_dro')

gpath = "../generated-data/ppc_yeast"
gi_task_path = "../generated-data/task_yeast_gi_costanzo"
#tasks.yeast_gi_costanzo.main(gpath, [26], [(0, 0), (0, 1), (1, 0), (1, 1)], gi_task_path)
utils.bin_simple.main(gi_task_path)
utils.bin_interacting.main(gi_task_path)
#utils.cv_gi.main(gi_task_path, 10, 4, 0.2)

# gpath = "../generated-data/ppc_yeast"
# gi_task_path = "../generated-data/task_yeast_gi_hybrid"
# tasks.yeast_gi_hybrid.main(gpath, '../generated-data/biogrid_yeast', '../generated-data/costanzo_gi', gi_task_path)
# utils.bin_simple.main(gi_task_path)
# utils.bin_interacting.main(gi_task_path)
# utils.cv_gi.main(gi_task_path, 10, 4, 0.2)

# gpath = "../generated-data/ppc_pombe"
# gi_task_path = "../generated-data/task_pombe_gi"
# smf_binned_path = "../generated-data/features/ppc_pombe_smf_binned.npz"
# tasks.pombe_gi.main(gpath, "../generated-data/biogrid_pombe", smf_binned_path, gi_task_path)
# utils.bin_simple.main(gi_task_path)
# utils.bin_interacting.main(gi_task_path)
# utils.cv_gi.main(gi_task_path, 10, 4, 0.2)

# gpath = "../generated-data/ppc_human"
# gi_task_path = "../generated-data/task_human_gi"
# smf_binned_path = "../generated-data/features/ppc_human_smf_binned.npz"
# #tasks.biogrid_plus_negative_sampling.main(gpath, "../generated-data/biogrid_human", smf_binned_path, gi_task_path)
# tasks.biogrid_plus_negative_sampling.main(gpath, "../generated-data/biogrid_human", smf_binned_path, gi_task_path + "_unfiltered", with_smf_only=False)
# #utils.bin_simple.main(gi_task_path)
# #utils.bin_interacting.main(gi_task_path)
# #utils.cv_gi.main(gi_task_path, 10, 4, 0.2)

# gpath = "../generated-data/ppc_dro"
# gi_task_path = "../generated-data/task_dro_gi"
# smf_binned_path = "../generated-data/features/ppc_dro_smf_binned.npz"
# #tasks.biogrid_plus_negative_sampling.main(gpath, "../generated-data/fb_dro", smf_binned_path, gi_task_path)
# tasks.biogrid_plus_negative_sampling.main(gpath, "../generated-data/fb_dro", smf_binned_path, gi_task_path + "_unfiltered", with_smf_only=False)
# # utils.bin_simple.main(gi_task_path)
# # utils.bin_interacting.main(gi_task_path)
# # utils.cv_gi.main(gi_task_path, 10, 4, 0.2)
