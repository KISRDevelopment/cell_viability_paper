import models.gi_mn
import json 
import os 
import numpy as np

# create GO without some terms
# exclude_GOIDS = ['GO:0016791', 'GO:0016301', 'GO:0008134']
# d = np.load("../generated-data/features/ppc_yeast_common_sgo.npz")
# labels = d['feature_labels']

# ix = [i for i in range(d['F'].shape[1]) if labels[i] not in exclude_GOIDS]

# F = d['F'][:,ix]
# labels = labels[ix]

# np.savez("../generated-data/features/ppc_yeast_common_sgo_strict.npz", F=F, feature_labels=labels)


# os.makedirs("../results/gi_cross_pred/", exist_ok=True)

# cfg_path = 'cfgs/models/yeast_gi_mn_full.json'
# with open(cfg_path, 'r') as f:
#     cfg = json.load(f)

# cfg['epochs'] = 1000
# models.gi_mn.main(cfg, 0, 0, "../tmp/test")

# cols = ["is_neutral", "rel_not_ppc", "rel_not_phospho", "rel_not_trans", "rel_not_complex", "rel_not_pathway"]

# for col in cols:
#     print("Column: %s" % col)
#     cfg_path = 'cfgs/models/yeast_gi_mn_full.json'
#     with open(cfg_path, 'r') as f:
#         cfg = json.load(f)


#     cfg['test_on_full_dataset'] = True

#     cfg['target_col'] = col 
#     cfg['train_model'] = False
    
#     models.gi_mn.main(cfg, 0, 0, "../results/gi_cross_pred/%s" % col)


os.makedirs("../results/gi_cross_pred_no_topology/", exist_ok=True)

cfg_path = 'cfgs/models/yeast_gi_mn_no_topology.json'
with open(cfg_path, 'r') as f:
    cfg = json.load(f)

cfg['epochs'] = 1000
models.gi_mn.main(cfg, 0, 0, "../tmp/test")

cols = ["is_neutral", "rel_not_ppc", "rel_not_phospho", "rel_not_trans", "rel_not_complex", "rel_not_pathway"]

for col in cols:
    print("Column: %s" % col)
    cfg_path = 'cfgs/models/yeast_gi_mn_no_topology.json'
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)


    cfg['test_on_full_dataset'] = True

    cfg['target_col'] = col 
    cfg['train_model'] = False
    
    models.gi_mn.main(cfg, 0, 0, "../results/gi_cross_pred_no_topology/%s" % col)


