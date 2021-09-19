import os 
import subprocess 
import sys 
import json 
import shlex

import models.null_model
import models.smf_nn
import models.smf_ordinal

import utils.union_sgo_terms

import feature_preprocessing.sgo 
import models.ensemble_model
import utils.eval_funcs
import analysis.fig_cv_performance

n_models = 10
go_postfix = "common_sgo"
output_dir = "../results/smf_generalization"
splits = [(i, 0) for i in range(n_models)]

# 1. Train yeast refined model on full yeast dataset and evaluate on the other three

refined_cfg_path = "cfgs/models/yeast_smf_refined_model.json"
with open(refined_cfg_path, 'r') as f:
    refined_cfg = json.load(f)
refined_cfg['train_on_full_dataset'] = True 
refined_cfg['epochs'] = 10000
refined_cfg['train_model'] = True 
refined_cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_yeast_%s.npz' % go_postfix]
os.makedirs('%s/refined_model' % output_dir, exist_ok=True)

model_files = []
for r, f in splits:
    refined_cfg['trained_model_path'] = '%s/refined_model/%d_%d' % (output_dir, r,f)
    model_files.append(refined_cfg['trained_model_path'])
    models.smf_nn.main(refined_cfg, r, f, '../tmp/dummy')

pombe_cfg_path = 'cfgs/models/pombe_smf_refined_model.json'
with open(pombe_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_pombe_%s.npz' % go_postfix]
models.ensemble_model.main(models.smf_nn, cfg, model_files, "%s/pombe_refined" % output_dir)

human_cfg_path = 'cfgs/models/human_smf_refined_model.json'
with open(human_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_human_%s.npz' % go_postfix]
models.ensemble_model.main(models.smf_nn, cfg, model_files, "%s/human_refined" % output_dir)

dro_cfg_path = 'cfgs/models/dro_smf_refined_model.json'
with open(dro_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_dro_%s.npz' % go_postfix]
models.ensemble_model.main(models.smf_nn, cfg, model_files, "%s/dro_refined" % output_dir)

# 2. Train OR model
or_cfg_path = "cfgs/models/yeast_smf_orm.json"
with open(or_cfg_path, 'r') as f:
    or_cfg = json.load(f)
or_cfg['train_on_full_dataset'] = True 
or_cfg['train_model'] = True 
or_cfg['spec'][2]['path'] = '../generated-data/features/ppc_yeast_%s.npz' % go_postfix
os.makedirs('%s/orm' % output_dir, exist_ok=True)
splits = [(i, 0) for i in range(10)]
model_files = []
for r, f in splits:
    or_cfg['trained_model_path'] = '%s/orm/%d_%d' % (output_dir, r,f)
    model_files.append(or_cfg['trained_model_path'])
    models.smf_ordinal.main(or_cfg, r, f, '../tmp/dummy')

pombe_cfg_path = 'cfgs/models/pombe_smf_orm.json'
with open(pombe_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][2]['path'] = '../generated-data/features/ppc_pombe_%s.npz' % go_postfix
models.ensemble_model.main(models.smf_ordinal, cfg, model_files, "%s/pombe_orm" % output_dir)

human_cfg_path = 'cfgs/models/human_smf_orm.json'
with open(human_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][2]['path'] = '../generated-data/features/ppc_human_%s.npz' % go_postfix
models.ensemble_model.main(models.smf_ordinal, cfg, model_files, "%s/human_orm" % output_dir)

dro_cfg_path = 'cfgs/models/dro_smf_orm.json'
with open(dro_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][2]['path'] = '../generated-data/features/ppc_dro_%s.npz' % go_postfix
models.ensemble_model.main(models.smf_ordinal, cfg, model_files, "%s/dro_orm" % output_dir)

# 3. Train scrambled Null model
refined_cfg_path = "cfgs/models/yeast_smf_refined_model.json"
with open(refined_cfg_path, 'r') as f:
    refined_cfg = json.load(f)
refined_cfg['scramble'] = True
refined_cfg['train_on_full_dataset'] = True 
refined_cfg['train_model'] = True 
refined_cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_yeast_%s.npz' % go_postfix]
os.makedirs('%s/null_scrambled' % output_dir, exist_ok=True)

splits = [(i, 0) for i in range(10)]
model_files = []
for r, f in splits:
    refined_cfg['trained_model_path'] = '%s/null_scrambled/%d_%d' % (output_dir, r,f)
    model_files.append(refined_cfg['trained_model_path'])
    models.smf_nn.main(refined_cfg, r, f, '../tmp/dummy')

pombe_cfg_path = 'cfgs/models/pombe_smf_refined_model.json'
with open(pombe_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_pombe_%s.npz' % go_postfix]
models.ensemble_model.main(models.smf_nn, cfg, model_files, "%s/pombe_null_scrambled" % output_dir)

human_cfg_path = 'cfgs/models/human_smf_refined_model.json'
with open(human_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_human_%s.npz' % go_postfix]
models.ensemble_model.main(models.smf_nn, cfg, model_files, "%s/human_null_scrambled" % output_dir)

dro_cfg_path = 'cfgs/models/dro_smf_refined_model.json'
with open(dro_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_dro_%s.npz' % go_postfix]
models.ensemble_model.main(models.smf_nn, cfg, model_files, "%s/dro_null_scrambled" % output_dir)

# 4. Train null model
refined_cfg_path = "cfgs/models/yeast_smf_refined_model.json"
with open(refined_cfg_path, 'r') as f:
    refined_cfg = json.load(f)
refined_cfg['train_on_full_dataset'] = True 
refined_cfg['train_model'] = True 
refined_cfg["trained_model_path"]= "%s/null_model.npz" % output_dir
models.null_model.main(refined_cfg, 0, 0, "../tmp/dummy")

pombe_cfg_path = 'cfgs/models/pombe_smf_refined_model.json'
with open(pombe_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg["trained_model_path"] = refined_cfg["trained_model_path"]
models.null_model.main(cfg, 0, 0, "%s/pombe_null" % output_dir)

human_cfg_path = 'cfgs/models/human_smf_refined_model.json'
with open(human_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg["trained_model_path"] = refined_cfg["trained_model_path"]
models.null_model.main(cfg, 0, 0, "%s/human_null" % output_dir)


dro_cfg_path = 'cfgs/models/dro_smf_refined_model.json'
with open(dro_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg["trained_model_path"] = refined_cfg["trained_model_path"]
models.null_model.main(cfg, 0, 0, "%s/dro_null" % output_dir)

# Analysis
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_pombe_binary.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_human_binary.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_dro_binary.json")
