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

# 1. unify sGO terms
# unified_sgo_terms = utils.union_sgo_terms.get_union([
#     "../generated-data/features/ppc_yeast-sgo.npz", 
#     "../generated-data/features/ppc_pombe_sgo.npz",
#     "../generated-data/features/ppc_human_sgo.npz",
#     "../generated-data/features/ppc_dro_sgo.npz"
# ])
# with open("../tmp/unified_sgo_terms", "w") as f:
#     f.write("\n".join(unified_sgo_terms))

# subprocess.call(shlex.split("../tools/owltools ../tools/go.obo --gaf ../data-sources/dro/fb.gaf --map2slim --idfile ../tmp/unified_sgo_terms --write-gaf ../tmp/dro.unified.sgo.gaf"))
# feature_preprocessing.sgo.main("../generated-data/ppc_dro", "../tmp/dro.unified.sgo.gaf", 1, output_path="../generated-data/features/ppc_dro_sgo_unified", all_go_terms=unified_sgo_terms)

# subprocess.call(shlex.split("../tools/owltools ../tools/go.obo --gaf ../data-sources/pombe/pombase.gaf --map2slim --idfile ../tmp/unified_sgo_terms --write-gaf ../tmp/pombase.unified.sgo.gaf"))
# feature_preprocessing.sgo.main("../generated-data/ppc_pombe", "../tmp/pombase.unified.sgo.gaf", 1, output_path="../generated-data/features/ppc_pombe_sgo_unified", all_go_terms=unified_sgo_terms)

# subprocess.call(shlex.split("../tools/owltools ../tools/go.obo --gaf ../data-sources/human/goa_human.gaf --map2slim --idfile ../tmp/unified_sgo_terms --write-gaf ../tmp/human.unified.sgo.gaf"))
# feature_preprocessing.sgo.main("../generated-data/ppc_human", "../tmp/human.unified.sgo.gaf", 2, output_path="../generated-data/features/ppc_human_sgo_unified", all_go_terms=unified_sgo_terms)

# subprocess.call(shlex.split("../tools/owltools ../tools/go.obo --gaf ../data-sources/yeast/sgd.gaf --map2slim --idfile ../tmp/unified_sgo_terms --write-gaf ../tmp/yeast.unified.sgo.gaf"))
# feature_preprocessing.sgo.main("../generated-data/ppc_yeast", "../tmp/yeast.unified.sgo.gaf", 2, output_path="../generated-data/features/ppc_yeast_sgo_unified", all_go_terms=unified_sgo_terms, annotations_reader=feature_preprocessing.sgo.read_annotations_yeast)


# 1. Train yeast refined model on full yeast dataset and evaluate on the other three

refined_cfg_path = "cfgs/models/yeast_smf_refined_model.json"
with open(refined_cfg_path, 'r') as f:
    refined_cfg = json.load(f)
refined_cfg['train_on_full_dataset'] = True 
refined_cfg['train_model'] = True 
refined_cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_yeast_sgo_unified.npz']
os.makedirs('../results/smf_generalization/refined_model', exist_ok=True)

splits = [(i, 0) for i in range(10)]
model_files = []
for r, f in splits:
    refined_cfg['trained_model_path'] = '../results/smf_generalization/refined_model/%d_%d' % (r,f)
    model_files.append(refined_cfg['trained_model_path'])
    models.smf_nn.main(refined_cfg, r, f, '../tmp/dummy')

pombe_cfg_path = 'cfgs/models/pombe_smf_refined_model.json'
with open(pombe_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_pombe_sgo_unified.npz']
models.ensemble_model.main(models.smf_nn, cfg, model_files, "../results/smf_generalization/pombe_refined")

human_cfg_path = 'cfgs/models/human_smf_refined_model.json'
with open(human_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_human_sgo_unified.npz']
models.ensemble_model.main(models.smf_nn, cfg, model_files, "../results/smf_generalization/human_refined")

dro_cfg_path = 'cfgs/models/dro_smf_refined_model.json'
with open(dro_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_dro_sgo_unified.npz']
models.ensemble_model.main(models.smf_nn, cfg, model_files, "../results/smf_generalization/dro_refined")

# 2. Train OR model
or_cfg_path = "cfgs/models/yeast_smf_orm.json"
with open(or_cfg_path, 'r') as f:
    or_cfg = json.load(f)
or_cfg['train_on_full_dataset'] = True 
or_cfg['train_model'] = True 
or_cfg['spec'][2]['path'] = '../generated-data/features/ppc_yeast_sgo_unified.npz'
os.makedirs('../results/smf_generalization/orm', exist_ok=True)
splits = [(i, 0) for i in range(10)]
model_files = []
for r, f in splits:
    or_cfg['trained_model_path'] = '../results/smf_generalization/orm/%d_%d' % (r,f)
    model_files.append(or_cfg['trained_model_path'])
    models.smf_ordinal.main(or_cfg, r, f, '../tmp/dummy')

pombe_cfg_path = 'cfgs/models/pombe_smf_orm.json'
with open(pombe_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][2]['path'] = '../generated-data/features/ppc_pombe_sgo_unified.npz'
models.ensemble_model.main(models.smf_ordinal, cfg, model_files, "../results/smf_generalization/pombe_orm")

human_cfg_path = 'cfgs/models/human_smf_orm.json'
with open(human_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][2]['path'] = '../generated-data/features/ppc_human_sgo_unified.npz'
models.ensemble_model.main(models.smf_ordinal, cfg, model_files, "../results/smf_generalization/human_orm")

dro_cfg_path = 'cfgs/models/dro_smf_orm.json'
with open(dro_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][2]['path'] = '../generated-data/features/ppc_dro_sgo_unified.npz'
models.ensemble_model.main(models.smf_ordinal, cfg, model_files, "../results/smf_generalization/dro_orm")

# 3. Train scrambled Null model
refined_cfg_path = "cfgs/models/yeast_smf_refined_model.json"
with open(refined_cfg_path, 'r') as f:
    refined_cfg = json.load(f)
refined_cfg['scramble'] = True
refined_cfg['train_on_full_dataset'] = True 
refined_cfg['train_model'] = True 
refined_cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_yeast_sgo_unified.npz']
os.makedirs('../results/smf_generalization/null_scrambled', exist_ok=True)

splits = [(i, 0) for i in range(10)]
model_files = []
for r, f in splits:
    refined_cfg['trained_model_path'] = '../results/smf_generalization/null_scrambled/%d_%d' % (r,f)
    model_files.append(refined_cfg['trained_model_path'])
    models.smf_nn.main(refined_cfg, r, f, '../tmp/dummy')

pombe_cfg_path = 'cfgs/models/pombe_smf_refined_model.json'
with open(pombe_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_pombe_sgo_unified.npz']
models.ensemble_model.main(models.smf_nn, cfg, model_files, "../results/smf_generalization/pombe_null_scrambled")

human_cfg_path = 'cfgs/models/human_smf_refined_model.json'
with open(human_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_human_sgo_unified.npz']
models.ensemble_model.main(models.smf_nn, cfg, model_files, "../results/smf_generalization/human_null_scrambled")

dro_cfg_path = 'cfgs/models/dro_smf_refined_model.json'
with open(dro_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_dro_sgo_unified.npz']
models.ensemble_model.main(models.smf_nn, cfg, model_files, "../results/smf_generalization/dro_null_scrambled")

# 4. Train null model
refined_cfg_path = "cfgs/models/yeast_smf_refined_model.json"
with open(refined_cfg_path, 'r') as f:
    refined_cfg = json.load(f)
refined_cfg['train_on_full_dataset'] = True 
refined_cfg['train_model'] = True 
refined_cfg["trained_model_path"]= "../results/smf_generalization/null_model.npz"
models.null_model.main(refined_cfg, 0, 0, "../tmp/dummy")

pombe_cfg_path = 'cfgs/models/pombe_smf_refined_model.json'
with open(pombe_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg["trained_model_path"] = refined_cfg["trained_model_path"]
models.null_model.main(cfg, 0, 0, "../results/smf_generalization/pombe_null")

human_cfg_path = 'cfgs/models/human_smf_refined_model.json'
with open(human_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg["trained_model_path"] = refined_cfg["trained_model_path"]
models.null_model.main(cfg, 0, 0, "../results/smf_generalization/human_null")


dro_cfg_path = 'cfgs/models/dro_smf_refined_model.json'
with open(dro_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['test_on_full_dataset'] = True
cfg["trained_model_path"] = refined_cfg["trained_model_path"]
models.null_model.main(cfg, 0, 0, "../results/smf_generalization/dro_null")

# Analysis
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_pombe.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_human.json")
analysis.fig_cv_performance.main("cfgs/fig_cv_performance/generalization_smf_dro.json")
