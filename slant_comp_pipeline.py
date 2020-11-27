import numpy as np 
import pandas as pd 
import tasks.slant_prediction_db
import models.gi_mn
import json

def load_cfg(path, model_path, remove_specs=[], **kwargs):
    with open(path, 'r') as f:
        cfg = json.load(f)

    cfg['bootstrap_training'] = False 
    cfg['early_stopping'] = False 
    cfg['train_on_full_dataset'] = False
    cfg['train_model'] = True 
    cfg['trained_model_path'] = model_path
    cfg['save_tjs'] = False 
    cfg['balanced_loss'] = True 

    cfg['spec'] = [s for s in cfg['spec'] if s['name'] not in remove_specs]
    cfg.update(kwargs)

    return cfg 

def predict(model, processors, path):

    df = pd.read_csv(path)
    ix = df['a_id'] != df['b_id']
    df = df[ix]

    features = []
    for proc in processors:
        features.append(proc.transform(df))
    if len(processors) > 1:
        batch_F = np.hstack(features)
    else:
        batch_F = features      
    preds = model.predict(batch_F, batch_size=10000)

    df['pred'] = preds[:,0]

    return df 

def sweep_thresholds(df, quality):
    ix_positives = df['source'].str.contains('BioGRID') & (df['quality'] >= quality)

    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        ix_pp = df['pred'] > t
        ix_tp = ix_positives & ix_pp

        print("Threshold %0.2f" % t)
        print("Positives in BioGRID: %d" % (np.sum(ix_positives)))
        print("Predicted positive: %d" % np.sum(ix_pp))
        print("Predicted positive & BioGRID: %d" % (np.sum(ix_tp)))
        print("TPR: %0.2f" % (np.sum(ix_tp) / np.sum(ix_positives)))
        print()

def analyze_slant_stats(df, quality):

    ix_positives = df['source'].str.contains('BioGRID') & (df['quality'] >= quality)
    ix_pp = df['source'].str.contains('Slorth')
    ix_tp = ix_positives & ix_pp

    print("SLANT Stats:")
    print("Dataset size: %d" % df.shape[0])
    print("Positives in BioGRID: %d" % (np.sum(ix_positives)))
    print("Predicted positive: %d" % np.sum(ix_pp))
    print("Predicted positive & BioGRID: %d" % (np.sum(ix_tp)))
    print("TPR: %0.2f" % (np.sum(ix_tp) / np.sum(ix_positives)))
    print()


mdl = models.gi_mn


# tasks.slant_prediction_db.main('yeast',
#     '../generated-data/ppc_yeast', 
#     '../data-sources/slant/s.cerevisiae_ssl_predictions.csv', 
#     '../generated-data/slant/yeast.csv')

# tasks.slant_prediction_db.main('pombe',
#     '../generated-data/ppc_pombe', 
#     '../data-sources/slant/s.pombe_ssl_predictions.csv', 
#     '../generated-data/slant/pombe.csv')

# tasks.slant_prediction_db.main('dro',
#     '../generated-data/ppc_dro', 
#     '../data-sources/slant/d.melanogaster_ssl_predictions.csv', 
#     '../generated-data/slant/dro.csv')

# tasks.slant_prediction_db.main('human',
#     '../generated-data/ppc_human', 
#     '../data-sources/slant/h.sapiens_ssl_predictions.csv', 
#     '../generated-data/slant/human.csv')

#
# YEAST
#

ycfg = load_cfg("cfgs/models/yeast_gi_mn.json", 
    "../results/models/yeast_gi_mn",
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz",
    epochs=50)

# 1. train model
mdl.main(ycfg, 0, 0, '../tmp/dummy')

# 2. load model
ycfg['train_model'] = False
model, processors = mdl.main(ycfg, 0, 0, '../tmp/dummy', return_model=True )

# 3. predict on slant db
df = predict(model, processors, '../generated-data/slant/yeast.csv')

# 4. analyze
sweep_thresholds(df, 1)
analyze_slant_stats(df, 1)

#
# Pombe
#
pcfg = load_cfg("cfgs/models/pombe_gi_mn.json",
    "../results/models/pombe_gi_mn", 
    targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz")

# 1. train model
mdl.main(pcfg, 0, 0, '../tmp/dummy')

# 2. load model
pcfg['train_model'] = False
model, processors = mdl.main(pcfg, 0, 0, '../tmp/dummy', return_model=True )

# 3. predict on slant db
df = predict(model, processors, '../generated-data/slant/pombe.csv')

# 4. analyze
sweep_thresholds(df, 1)
analyze_slant_stats(df, 1)


#
# Human
#
hcfg = load_cfg("cfgs/models/human_gi_mn.json",
    "../results/models/human_gi_mn", 
    targets_path="../generated-data/targets/task_human_gi_bin_interacting.npz")

# 1. train model
mdl.main(hcfg, 0, 0, '../tmp/dummy')

# 2. load model
hcfg['train_model'] = False
model, processors = mdl.main(hcfg, 0, 0, '../tmp/dummy', return_model=True )

# 3. predict on slant db
df = predict(model, processors, '../generated-data/slant/human.csv')

# 4. analyze
sweep_thresholds(df, 1)
analyze_slant_stats(df, 1)
