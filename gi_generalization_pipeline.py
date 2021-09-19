import os 
import subprocess 
import sys 
import json 
import shlex

import models.null_model
import models.gi_nn
import models.gi_mn
import models.ensemble_model
import utils.eval_funcs
import analysis.fig_cv_performance
n_models = 10
splits = [(i, 0) for i in range(n_models)]

def generalize(yeast_cfg, mdl, org_cfgs, output_dir):
    
    cfg_name = yeast_cfg['name']
    model_files_dir = os.path.join(output_dir, cfg_name)
    os.makedirs(model_files_dir, exist_ok=True)

    yeast_cfg['train_on_full_dataset'] = True 
    yeast_cfg['train_model'] = True
    
    # train source models
    model_files = []
    for r, f in splits:
        yeast_cfg['trained_model_path'] = os.path.join(model_files_dir, "%d_%d" % (r,f))
        print(yeast_cfg['trained_model_path'])
        model_files.append(yeast_cfg['trained_model_path'])
        mdl.main(yeast_cfg, r, f, '../tmp/dummy')

    # test on target models
    for cfg in org_cfgs:
        cfg_name = cfg['name']
        cfg['train_model'] = False
        cfg['test_on_full_dataset'] = True
        models.ensemble_model.main(mdl, cfg, model_files, os.path.join(output_dir, cfg_name))

def load_cfg(path, remove_specs=[], **kwargs):
    with open(path, 'r') as f:
        cfg = json.load(f)

    cfg['spec'] = [s for s in cfg['spec'] if s['name'] not in remove_specs]
    cfg.update(kwargs)
    print(cfg)
    return cfg 

#refined model
# yeast_cfg = load_cfg("cfgs/models/yeast_gi_refined_model.json",
#     task_path="../generated-data/task_yeast_gi_hybrid",
#     splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
#     target_col = "is_neutral",
#     name="yeast_refined")
# pombe_cfg = load_cfg("cfgs/models/pombe_gi_refined_model.json", 
#     target_col="is_neutral",
#     name="pombe_refined")
# human_cfg = load_cfg("cfgs/models/human_gi_refined_model.json", name="human_refined", target_col="is_neutral")
# dro_cfg = load_cfg("cfgs/models/dro_gi_refined_model.json", name="dro_refined", target_col="is_neutral")
# generalize(yeast_cfg, models.gi_nn, [pombe_cfg, human_cfg, dro_cfg],"../results/gi_generalization")

# multinomial model
# yeast_cfg = load_cfg("cfgs/models/yeast_gi_mn.json",
#     task_path="../generated-data/task_yeast_gi_hybrid",
#     splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
#     target_col = "is_neutral",
#     add_repfold_to_trained_model_path=False,
#     epochs=1000,
#     name="yeast_mn")
# pombe_cfg = load_cfg("cfgs/models/pombe_gi_mn.json",
#     target_col="is_neutral", name="pombe_mn",add_repfold_to_trained_model_path=False)
# human_cfg = load_cfg("cfgs/models/human_gi_mn.json", name="human_mn", target_col="is_neutral",add_repfold_to_trained_model_path=False)
# dro_cfg = load_cfg("cfgs/models/dro_gi_mn.json", name="dro_mn", target_col="is_neutral",add_repfold_to_trained_model_path=False)
# generalize(yeast_cfg, models.gi_mn, [pombe_cfg, human_cfg, dro_cfg],"../results/gi_generalization")

# multinomial model - no sgo no smf
# yeast_cfg = load_cfg("cfgs/models/yeast_gi_mn.json",
#     task_path="../generated-data/task_yeast_gi_hybrid",
#     splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
#     target_col = "is_neutral",
#     remove_specs=["sgo", "smf"],
#     add_repfold_to_trained_model_path=False,
#     epochs=1000,
#     name="yeast_mn_no_sgo_smf")
# pombe_cfg = load_cfg("cfgs/models/pombe_gi_mn.json",
#     remove_specs=["sgo", "smf"],
#     target_col="is_neutral", name="pombe_mn_no_sgo_smf",add_repfold_to_trained_model_path=False)
# human_cfg = load_cfg("cfgs/models/human_gi_mn.json", name="human_mn_no_sgo_smf", target_col="is_neutral",add_repfold_to_trained_model_path=False,remove_specs=["sgo", "smf"])
# dro_cfg = load_cfg("cfgs/models/dro_gi_mn.json", name="dro_mn_no_sgo_smf", target_col="is_neutral",add_repfold_to_trained_model_path=False,remove_specs=["sgo", "smf"])
#generalize(yeast_cfg, models.gi_mn, [pombe_cfg, human_cfg, dro_cfg],"../results/gi_generalization")

yeast_cfg = load_cfg("cfgs/models/yeast_gi_mn.json",
    task_path="../generated-data/task_yeast_gi_hybrid",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
    target_col = "is_neutral",
    remove_specs=["sgo"],
    add_repfold_to_trained_model_path=False,
    epochs=1000,
    name="yeast_mn_no_sgo")
pombe_cfg = load_cfg("cfgs/models/pombe_gi_mn.json",
    remove_specs=["sgo"],
    target_col="is_neutral", name="pombe_mn_no_sgo",add_repfold_to_trained_model_path=False)
human_cfg = load_cfg("cfgs/models/human_gi_mn.json", name="human_mn_no_sgo", target_col="is_neutral",
    add_repfold_to_trained_model_path=False,remove_specs=["sgo"])
dro_cfg = load_cfg("cfgs/models/dro_gi_mn.json", name="dro_mn_no_sgo", target_col="is_neutral",add_repfold_to_trained_model_path=False,
    remove_specs=["sgo"])
generalize(yeast_cfg, models.gi_mn, [pombe_cfg, human_cfg, dro_cfg],"../results/gi_generalization")



# #multinomial model
# yeast_cfg = load_cfg("cfgs/models/yeast_gi_mn.json",
#     targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz", 
#     remove_specs=["sgo", "smf"],
#     epochs=1000,
#     name="yeast_mn_no_sgo_smf")
# pombe_cfg = load_cfg("cfgs/models/pombe_gi_mn.json",
    
#     remove_specs=["sgo", "smf"],
#     targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz", name="pombe_mn_no_sgo_smf")
# human_cfg = load_cfg("cfgs/models/human_gi_mn.json", name="human_mn_no_sgo_smf", 
#     remove_specs=["sgo", "smf"])
# dro_cfg = load_cfg("cfgs/models/dro_gi_mn.json", name="dro_mn_no_sgo_smf", 
#     remove_specs=["sgo", "smf"])
# generalize(yeast_cfg, models.gi_mn, [pombe_cfg, human_cfg, dro_cfg],"../results/gi_generalization")

#null
yeast_cfg = load_cfg("cfgs/models/yeast_gi_mn.json",
    task_path="../generated-data/task_yeast_gi_hybrid",
    splits_path="../generated-data/splits/task_yeast_gi_hybrid_full.npz",
    target_col = "is_neutral",
    add_repfold_to_trained_model_path=False,
    epochs=1000,
    name="yeast_null")
pombe_cfg = load_cfg("cfgs/models/pombe_gi_mn.json",
    target_col="is_neutral", name="pombe_null",add_repfold_to_trained_model_path=False)
human_cfg = load_cfg("cfgs/models/human_gi_mn.json", name="human_null", target_col="is_neutral",add_repfold_to_trained_model_path=False)
dro_cfg = load_cfg("cfgs/models/dro_gi_mn.json", name="dro_null", target_col="is_neutral",add_repfold_to_trained_model_path=False)
#generalize(yeast_cfg, models.null_model, [pombe_cfg, human_cfg, dro_cfg],"../results/gi_generalization")

# Analysis
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_gi_pombe.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_gi_human.json")
# analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_gi_dro.json")
