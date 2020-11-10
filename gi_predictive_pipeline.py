import os 
import sys 
import json 
import models.gi_mn
import networkx as nx 
import itertools 
import numpy as np 
import pandas as pd 

if not os.path.isdir("../results/models"):
    os.mkdir("../results/models")
if not os.path.isdir("../results/models_tjs"):
    os.mkdir("../results/models_tjs")

def load_cfg(path, model_path, tjs_model_path, **kwargs):
    with open(path, 'r') as f:
        cfg = json.load(f)

    cfg['bootstrap_training'] = False 
    cfg['early_stopping'] = False 
    cfg['train_on_full_dataset'] = True
    cfg['train_model'] = True 
    cfg['epochs'] = 50
    cfg['trained_model_path'] = model_path
    cfg['save_tjs'] = True 
    cfg['tjs_path'] = tjs_model_path

    cfg.update(kwargs)

    return cfg 

mdl = models.gi_mn 

#
# Train Species Models
#
yeast_cfg = load_cfg("cfgs/models/yeast_gi_mn.json",
    "../results/models/yeast_gi_mn", 
    "../results/models_tjs/yeast_gi_mn",
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz")

pombe_cfg = load_cfg("cfgs/models/pombe_gi_mn.json",
    "../results/models/pombe_gi_mn", 
    "../results/models_tjs/pombe_gi_mn",
    targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz")

human_cfg = load_cfg("cfgs/models/human_gi_mn.json",
    "../results/models/human_gi_mn", 
    "../results/models_tjs/human_gi_mn")

dro_cfg = load_cfg("cfgs/models/dro_gi_mn.json",
    "../results/models/dro_gi_mn", 
    "../results/models_tjs/dro_gi_mn")

# mdl.main(yeast_cfg, 0, 0, '../tmp/dummy')
# mdl.main(pombe_cfg, 0, 0, '../tmp/dummy')
# mdl.main(human_cfg, 0, 0, '../tmp/dummy')
# mdl.main(dro_cfg, 0, 0, '../tmp/dummy')

def generate_predictions(cfg, gpath, result_path):

    main_df = pd.read_csv(cfg['task_path'])
    interacting_df = main_df[main_df['bin'] != 1]
    interacting_pairs = set('%d,%d' % tuple(sorted((a,b))) for a,b in zip(interacting_df['a_id'], interacting_df['b_id']))
    print("reported pairs: %d" % len(interacting_pairs))
    
    cfg['train_model']= False
    model, processors = mdl.main(cfg, 0, 0, '../tmp/dummy', return_model=True )

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())

    rows = []
    BATCH_SIZE = 250000
    stored = 0
    i = 0
    n_total = (len(nodes) * (len(nodes) - 1)) / 2

    first_append = True
    for comb in itertools.combinations(np.arange(len(nodes)), 2):
        rows.append({ 'a_id' : comb[0], 'b_id' : comb[1] })
        i += 1
        if len(rows) == BATCH_SIZE or i == n_total:
            df = pd.DataFrame(rows)
            df['pair'] = ['%d,%d' % t for t in zip(df['a_id'], df['b_id'])]
            df['reported_gi'] =  df['pair'].isin(interacting_pairs).astype(int)

            rows = []

            features = []
            for proc in processors:
                features.append(proc.transform(df))
            
            batch_F = np.hstack(features)

            preds = model.predict(batch_F, batch_size=BATCH_SIZE)
            preds_hard = np.argmax(preds, axis=1)

            ix = (preds_hard == 0) | df['reported_gi']
            df['prob_gi'] = preds[:,0]
            
            output_df = df[ix][['a_id', 'b_id', 'prob_gi', 'reported_gi']]

            
            output_df.to_csv(result_path, mode='w' if first_append else 'a', header=first_append, index=False)
            first_append = False 

            stored += output_df.shape[0]
            print("Completed %8.4f, num interactions=%d" % (i / n_total, stored))



#generate_predictions(yeast_cfg, '../generated-data/ppc_yeast', '../results/yeast_gi_preds')
#generate_predictions(pombe_cfg, '../generated-data/ppc_pombe', '../results/pombe_gi_preds')
#generate_predictions(dro_cfg, '../generated-data/ppc_dro', '../results/dro_gi_preds')
#generate_predictions(human_cfg, '../generated-data/ppc_human', '../results/human_gi_preds')

def select_subset(gpath, result_path, gene_names, output_path):
    gene_names = set(gene_names)

    df = pd.read_csv(result_path)
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    
    df['gene A'] = [nodes[a] for a in df['a_id']]
    df['gene B'] = [nodes[b] for b in df['b_id']]

    ix = df['gene A'].isin(gene_names) | df['gene B'].isin(gene_names)

    df = df[ix]
    print(df)
    df.to_csv(output_path, columns=['gene A', 'gene B', 'prob_gi', 'reported_gi'], index=False)

#select_subset('../generated-data/ppc_human', '../results/human_gi_preds', ['myc', 'tp53'], '../results/human_gi_preds_subset')
select_subset('../generated-data/ppc_dro', '../results/dro_gi_preds', ['fbgn0003366', 'fbgn0024248'], '../results/dro_gi_preds_subset')
