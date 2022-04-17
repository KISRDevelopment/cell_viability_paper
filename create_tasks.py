import os 
import pandas as pd
import utils.bin_outcomes

import tasks.yeast_smf
import tasks.pombe_smf
import tasks.human_smf
import tasks.human_smf_ca_mo_v
import tasks.dro_smf
import tasks.dro_smf_ca_mo_v

import pretasks.make_costanzo_dataset
import pretasks.make_biogrid_dataset
import pretasks.make_fb_dataset

import tasks.yeast_gi_costanzo
import tasks.yeast_gi_hybrid
import tasks.pombe_gi
import tasks.biogrid_plus_negative_sampling
import tasks.yeast_tgi

def main():
    os.makedirs("../generated-data", exist_ok=True)
    
    yeast_smf() 
    pombe_smf()
    human_smf()
    human_smf_ca_mo_v()
    dro_smf()
    dro_smf_ca_mo_v()

    gi_pretasks()

    yeast_gi_costanzo()
    yeast_gi_hybrid()

    pombe_gi()
    human_gi()
    dro_gi()
    yeast_tgi()

def add_smf_binary_outcomes(path):
    utils.bin_outcomes.main(path, {
        "is_lethal" : lambda bins: bins == 0,
        "is_viable" : lambda bins: bins > 0,
    }, path)

def yeast_smf():
    print("Creating Yeast SMF Prediction Task")
    
    gpath = "../generated-data/ppc_yeast"
    smf_task_path = "../generated-data/task_yeast_smf_30"
    tasks.yeast_smf.main(gpath, 30, smf_task_path)
    add_smf_binary_outcomes(smf_task_path)

def pombe_smf():
    print("Creating Fission Yeast SMF Prediction Task")
    
    gpath = "../generated-data/ppc_pombe"
    smf_task_path = "../generated-data/task_pombe_smf"

    tasks.pombe_smf.main(gpath, smf_task_path)
    add_smf_binary_outcomes(smf_task_path)

def human_smf():
    print("Creating Human SMF Prediction Task")

    gpath = "../generated-data/ppc_human"
    smf_task_path = "../generated-data/task_human_smf"
    tasks.human_smf.main(gpath, smf_task_path)
    add_smf_binary_outcomes(smf_task_path)

def human_smf_ca_mo_v():
    print("Creating Human CA vs MO vs V SMF Prediction Task")

    gpath = "../generated-data/ppc_human"
    cell_smf_task_path = "../generated-data/task_human_smf"
    smf_task_path = "../generated-data/task_human_smf_ca_mo_v"
    tasks.human_smf_ca_mo_v.main(gpath, cell_smf_task_path, smf_task_path)

    # remove the CA
    df = pd.read_csv(smf_task_path)
    df = df[df['bin'] > 0]
    df['bin'] = df['bin'] - 1

    df.to_csv("../generated-data/task_human_smf_mo_v", index=False)

def dro_smf():
    print("Creating Fruit Fly SMF Prediction Task")

    gpath = "../generated-data/ppc_dro"
    smf_task_path = "../generated-data/task_dro_smf"
    tasks.dro_smf.main(gpath, smf_task_path)
    add_smf_binary_outcomes(smf_task_path)

def dro_smf_ca_mo_v():
    print("Creating Fruit Fly CA vs MO vs V SMF Prediction Task")

    gpath = "../generated-data/ppc_dro"
    smf_task_path = "../generated-data/task_dro_smf_ca_mo_v"
    tasks.dro_smf_ca_mo_v.main(gpath, "../generated-data/task_dro_smf", smf_task_path)
    
    # remove the CA
    df = pd.read_csv(smf_task_path)
    df = df[df['bin'] > 0]
    df['bin'] = df['bin'] - 1

    df.to_csv("../generated-data/task_dro_smf_mo_v", index=False)

def gi_pretasks():
    print("Creating Pretask GI Datasets")
    
    pretasks.make_costanzo_dataset.main('../generated-data/pretask_costanzo_gi')
    pretasks.make_biogrid_dataset.main(559292, 3, '../generated-data/pretask_biogrid_yeast')
    pretasks.make_biogrid_dataset.main(284812, 2, '../generated-data/pretask_biogrid_pombe')
    pretasks.make_biogrid_dataset.main(9606, 1, '../generated-data/pretask_biogrid_human')
    pretasks.make_fb_dataset.main('../generated-data/pretask_fb_dro')

def yeast_gi_costanzo():
    print("Creating Yeast Costanzo GI Dataset")
    
    gpath = "../generated-data/ppc_yeast"
    pretask_path = "../generated-data/pretask_costanzo_gi"
    gi_task_path = "../generated-data/task_yeast_gi_costanzo"
    tasks.yeast_gi_costanzo.main(gpath, pretask_path, [26], [(0, 0), (0, 1), (1, 0), (1, 1)], gi_task_path, neg_thres=-0.08, pos_thres=0.08)

def yeast_gi_hybrid():
    print("Creating Yeast Hybrid GI Dataset")
    gpath = "../generated-data/ppc_yeast"
    gi_task_path = "../generated-data/task_yeast_gi_hybrid"
    tasks.yeast_gi_hybrid.main(gpath, 
        '../generated-data/pretask_biogrid_yeast', 
        '../generated-data/pretask_costanzo_gi', gi_task_path)
    utils.bin_outcomes.main(gi_task_path, {
        "is_negative" : lambda bins: bins == 0,
        "is_interacting" : lambda bins: bins != 1,
        "is_neutral" : lambda bins: bins == 1,
        "is_not_negative" : lambda bins: bins > 0
    }, gi_task_path)

def pombe_gi():
    print("Creating Fission Yeast GI Dataset")

    gpath = "../generated-data/ppc_pombe"
    gi_task_path = "../generated-data/task_pombe_gi"
    smf_task_path = "../generated-data/task_pombe_smf"
    tasks.pombe_gi.main(gpath, "../generated-data/pretask_biogrid_pombe", smf_task_path, gi_task_path)
    utils.bin_outcomes.main(gi_task_path, {
        "is_negative" : lambda bins: bins == 0,
        "is_interacting" : lambda bins: bins != 1,
        "is_neutral" : lambda bins: bins == 1
    }, gi_task_path)

def human_gi():
    print("Creating Human GI Dataset")

    gpath = "../generated-data/ppc_human"
    gi_task_path = "../generated-data/task_human_gi"
    smf_task_path = "../generated-data/task_human_smf"
    tasks.biogrid_plus_negative_sampling.main(gpath, 
        "../generated-data/pretask_biogrid_human", 
        smf_task_path, 
        gi_task_path, 
        with_smf_only=True)
    utils.bin_outcomes.main(gi_task_path, {
        "is_negative" : lambda bins: bins == 0,
        "is_interacting" : lambda bins: bins != 1,
        "is_neutral" : lambda bins: bins == 1
    }, gi_task_path)

def dro_gi():
    print("Creating Fruit Fly GI Dataset")

    gpath = "../generated-data/ppc_dro"
    gi_task_path = "../generated-data/task_dro_gi"
    smf_task_path = "../generated-data/task_dro_smf"
    tasks.biogrid_plus_negative_sampling.main(gpath, 
        "../generated-data/pretask_fb_dro", 
        smf_task_path, 
        gi_task_path, 
        with_smf_only=True)
    utils.bin_outcomes.main(gi_task_path, {
        "is_negative" : lambda bins: bins == 0,
        "is_interacting" : lambda bins: bins != 1,
        "is_neutral" : lambda bins: bins == 1
    }, gi_task_path)

def yeast_tgi():
    print("Creating Yeast Triple GI Dataset")

    gpath = "../generated-data/ppc_yeast"
    tgi_task_path = "../generated-data/task_yeast_tgi"
    tasks.yeast_tgi.main(gpath, tgi_task_path)

if __name__ == "__main__":
    main()
