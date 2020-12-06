#
# Trains models on the entire species datasets
#
import numpy as np 
import pandas as pd 
import models.gi_mn
import models.gi_nn
import utils.cv_simple
import json

def load_cfg(path, model_path, tjs_model_path="", remove_specs=[], **kwargs):
    with open(path, 'r') as f:
        cfg = json.load(f)

    cfg['bootstrap_training'] = False 
    cfg['early_stopping'] = True 
    cfg['train_on_full_dataset'] = True
    cfg['train_model'] = True 
    cfg['trained_model_path'] = model_path
    cfg['save_tjs'] = tjs_model_path != "" 
    cfg['balanced_loss'] = True 
    cfg['tjs_path'] = tjs_model_path

    cfg['spec'] = [s for s in cfg['spec'] if s['name'] not in remove_specs]
    cfg.update(kwargs)

    return cfg 


# ycfg = load_cfg("cfgs/models/yeast_gi_refined_model.json", 
#     "../results/models/yeast_gi_refined",
#     targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz")
# #models.gi_nn.main(ycfg, 0, 0, '../tmp/dummy')

# pcfg = load_cfg("cfgs/models/pombe_gi_refined_model.json",
#     "../results/models/pombe_gi_refined", 
#     targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz")
# models.gi_nn.main(pcfg, 0, 0, '../tmp/dummy')

# hcfg = load_cfg("cfgs/models/human_gi_refined_model.json",
#     "../results/models/human_gi_refined", 
#     targets_path="../generated-data/targets/task_human_gi_bin_interacting.npz")
# #models.gi_nn.main(hcfg, 0, 0, '../tmp/dummy')

# dcfg = load_cfg("cfgs/models/dro_gi_refined_model.json",
#     "../results/models/dro_gi_refined", 
#     targets_path="../generated-data/targets/task_dro_gi_bin_interacting.npz")
# #models.gi_nn.main(dcfg, 0, 0, '../tmp/dummy')

ycfg = load_cfg("cfgs/models/yeast_gi_mn.json", 
    "../results/models/yeast_gi_costanzo_mn",
    tjs_model_path="../results/models/tjs/yeast_gi_costanzo_mn",
    task_path = "../generated-data/task_yeast_gi_costanzo",
    splits_path = "../generated-data/splits/task_yeast_gi_costanzo_10reps_4folds_0.20valid.npz",
    targets_path="../generated-data/targets/task_yeast_gi_costanzo_bin_interacting.npz")
models.gi_mn.main(ycfg, 0, 0, '../tmp/dummy')

ycfg = load_cfg("cfgs/models/yeast_gi_mn.json", 
    "../results/models/yeast_gi_hybrid_mn",
    tjs_model_path="../results/models/tjs/yeast_gi_hybrid_mn",
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz")
models.gi_mn.main(ycfg, 0, 0, '../tmp/dummy')

pcfg = load_cfg("cfgs/models/pombe_gi_mn.json",
    "../results/models/pombe_gi_mn", 
    tjs_model_path="../results/models/tjs/pombe_gi_mn",
    targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz")
models.gi_mn.main(pcfg, 0, 0, '../tmp/dummy')

hcfg = load_cfg("cfgs/models/human_gi_mn.json",
    "../results/models/human_gi_mn", 
    tjs_model_path="../results/models/tjs/human_gi_mn",
    targets_path="../generated-data/targets/task_human_gi_bin_interacting.npz")
models.gi_mn.main(hcfg, 0, 0, '../tmp/dummy')

dcfg = load_cfg("cfgs/models/dro_gi_mn.json",
    "../results/models/dro_gi_mn", 
    tjs_model_path="../results/models/tjs/dro_gi_mn",
    targets_path="../generated-data/targets/task_dro_gi_bin_interacting.npz")
models.gi_mn.main(dcfg, 0, 0, '../tmp/dummy')
