import os 
import sys 
import json 
import models.gi_mn
import models.gi_nn
import networkx as nx 
import itertools 
import numpy as np 
import pandas as pd 
import sklearn.metrics 

def generate_predictions(mdl, cfg, result_path, thres=0.5, keep_net_preds=False, batch_size=250000):
    gpath = cfg['gpath']
    main_df = pd.read_csv(cfg['task_path'])

    interacting_df = main_df[main_df['bin'] != 1]
    interacting_pairs = set('%d,%d' % tuple(sorted((a,b))) for a,b in zip(interacting_df['a_id'], interacting_df['b_id']))
    print("interacting pairs: %d" % len(interacting_pairs))
    
    noninteracting_df = main_df[main_df['bin'] == 1]
    noninteracting_pairs = set('%d,%d' % tuple(sorted((a,b))) for a,b in zip(noninteracting_df['a_id'], noninteracting_df['b_id']))
    print("noninteracting pairs: %d" % len(noninteracting_pairs))

    training_ids = set(main_df['a_id']).union(main_df['b_id'])

    cfg['train_model']= False
    model, processors = mdl.main(cfg, 0, 0, '../tmp/dummy', return_model=True)

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    
    rows = []
    i = 0
    n_total = (len(nodes) * (len(nodes) - 1)) / 2

    first_append = True
    for comb in itertools.combinations(np.arange(len(nodes)), 2):
        rows.append({ 'a_id' : comb[0], 'b_id' : comb[1] })
        i += 1
        if len(rows) == batch_size or i == n_total:
            df = pd.DataFrame(rows)
            df['pair'] = ['%d,%d' % t for t in zip(df['a_id'], df['b_id'])]
            in_interacting = df['pair'].isin(interacting_pairs)
            in_noninteracting = df['pair'].isin(noninteracting_pairs)
            df['observed'] = (in_interacting | in_noninteracting).astype(int)
            df['interacting'] =  in_interacting.astype(int)
            df['novel'] = (~df['a_id'].isin(training_ids) & ~df['b_id'].isin(training_ids)).astype(int)

            rows = []

            features = []
            for proc in processors:
                features.append(proc.transform(df))
            
            if len(processors) > 1:
                batch_F = np.hstack(features)
            else:
                batch_F = features 
            
            preds = model.predict(batch_F, batch_size=batch_size)
            preds_gi = preds[:,0] > thres

            if not keep_net_preds:
                ix = (preds_gi == 1) | df['observed']
            else:
                ix = (preds_gi == 1) | (preds_gi == 0)
            df['prob_gi'] = preds[:,0]
            
            output_df = df[ix][['a_id', 'b_id', 'prob_gi', 'observed', 'interacting']]
            output_df['a_id'] = [nodes[e] for e in output_df['a_id']]
            output_df['b_id'] = [nodes[e] for e in output_df['b_id']]
            
            output_df.to_csv(result_path, mode='w' if first_append else 'a', header=first_append, index=False)
            first_append = False 

            print("Completed %8.4f" % (i / n_total))

def load_cfg(path, model_path, **kwargs):
    with open(path, 'r') as f:
        cfg = json.load(f)

    cfg['train_model'] = False 
    cfg['trained_model_path'] = model_path

    cfg.update(kwargs)

    return cfg 

def filter(path, thres):

    df = pd.read_csv(path)
    print("@ 0.5:")
    print_stats(df, 0.5)

    ix = (df['prob_gi'] >= thres) | (df['observed'] == 1) 
    df = df[ix]
    print("@ %0.2f" % thres)
    print_stats(df, thres)

    df.to_csv('%s_thres%0.2f' % (path, thres), index=False)
    
def print_stats(df, thres):
    print("Size: %d" % df.shape[0])

    ix_observed = df['observed'] == 1
    ix_observed_gis = ix_observed & (df['interacting'] == 1)
    
    ix_predicted_gis = df['prob_gi'] >= thres 
    ix_novel_gis = ~ix_observed & ix_predicted_gis
    print("Observations: %d, Observed GIs: %d, Predicted GIs: %d, Novel GIs: %d"
        % (np.sum(ix_observed), np.sum(ix_observed_gis), np.sum(ix_predicted_gis), np.sum(ix_novel_gis)))
    
    tpr = np.sum(ix_predicted_gis & ix_observed_gis) / np.sum(ix_observed_gis)
    fpr = np.sum(ix_predicted_gis & (df['interacting'] == 0) & ix_observed) / np.sum(ix_observed & (df['interacting'] == 0))

    print("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))
# cfg = load_cfg("cfgs/models/yeast_gi_mn.json", 
#     "../results/models/yeast_gi_costanzo_mn",
#     gpath = "../generated-data/ppc_yeast",
#     task_path = "../generated-data/task_yeast_gi_costanzo",
#     splits_path = "../generated-data/splits/task_yeast_gi_costanzo_10reps_4folds_0.20valid.npz",
#     targets_path="../generated-data/targets/task_yeast_gi_costanzo_bin_interacting.npz")
# generate_predictions(models.gi_mn, cfg, "../results/preds/yeast_gi_costanzo_mn")

# cfg = load_cfg("cfgs/models/yeast_gi_mn.json", 
#     "../results/models/yeast_gi_hybrid_mn",
#     gpath = "../generated-data/ppc_yeast",
#     targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz")
# generate_predictions(models.gi_mn, cfg, "../results/preds/yeast_gi_hybrid_mn")

# cfg = load_cfg("cfgs/models/pombe_gi_mn.json",
#     "../results/models/pombe_gi_mn", 
#     gpath = "../generated-data/ppc_pombe",
#     targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz")
# generate_predictions(models.gi_mn, cfg, "../results/preds/pombe_gi_mn")

# cfg = load_cfg("cfgs/models/human_gi_mn.json",
#     "../results/models/human_gi_mn", 
#     gpath = "../generated-data/ppc_human",
#     targets_path="../generated-data/targets/task_human_gi_bin_interacting.npz")
# generate_predictions(models.gi_mn, cfg, "../results/preds/human_gi_mn")

# cfg = load_cfg("cfgs/models/dro_gi_mn.json",
#     "../results/models/dro_gi_mn", 
#     gpath = "../generated-data/ppc_dro",
#     targets_path="../generated-data/targets/task_dro_gi_bin_interacting.npz")
# generate_predictions(models.gi_mn, cfg, "../results/preds/dro_gi_mn")

#filter("../results/preds/yeast_gi_costanzo_mn", 0.5)
#filter("../results/preds/yeast_gi_hybrid_mn", 0.65)
#filter("../results/preds/pombe_gi_mn", 0.55)
#filter("../results/preds/human_gi_mn", 0.75)
#filter("../results/preds/dro_gi_mn", 0.75)

filter("../results/preds/yeast_gi_costanzo_mn", 0.95)
filter("../results/preds/yeast_gi_hybrid_mn", 0.95)
filter("../results/preds/pombe_gi_mn", 0.95)
filter("../results/preds/human_gi_mn", 0.95)
filter("../results/preds/dro_gi_mn", 0.95)
