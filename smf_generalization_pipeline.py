import os 
import subprocess 
import sys 
import json 
import shlex

import models.smf_nn
import models.smf_ordinal

import utils.union_sgo_terms

import feature_preprocessing.sgo 

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


# 1. Train yeast refined model on full yeast dataset

refined_cfg_path = "cfgs/models/yeast_smf_refined_model.json"
with open(refined_cfg_path, 'r') as f:
    refined_cfg = json.load(f)
refined_cfg['train_on_full_dataset'] = True 
refined_cfg['train_model'] = True 
refined_cfg['trained_model_path'] = '../results/smf_generalization/refined'
refined_cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_yeast_sgo_unified.npz']
os.makedirs('../results/smf_generalization', exist_ok=True)
models.smf_nn.main(refined_cfg, 1, 0, "../tmp/dummy")

# 2. Test on Pombe
pombe_cfg_path = 'cfgs/models/pombe_smf_refined_model.json'
with open(pombe_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['train_model'] = False 
cfg['test_on_full_dataset'] = True
cfg['trained_model_path'] = refined_cfg['trained_model_path']
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_pombe_sgo_unified.npz']
models.smf_nn.main(cfg, 0, 0, "../tmp/test")

# 3. Test on  Human
human_cfg_path = 'cfgs/models/human_smf_refined_model.json'
with open(human_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['train_model'] = False 
cfg['test_on_full_dataset'] = True
cfg['trained_model_path'] = refined_cfg['trained_model_path']
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_human_sgo_unified.npz']
models.smf_nn.main(cfg, 0, 0, "../tmp/test")

# 4. Test on Dro
dro_cfg_path = 'cfgs/models/dro_smf_refined_model.json'
with open(dro_cfg_path, 'r') as f:
    cfg = json.load(f)
cfg['train_model'] = False 
cfg['test_on_full_dataset'] = True
cfg['trained_model_path'] = refined_cfg['trained_model_path']
cfg['spec'][1]['paths'] = ['../generated-data/features/ppc_dro_sgo_unified.npz']
models.smf_nn.main(cfg, 0, 0, "../tmp/test")