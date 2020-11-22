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

import tensorflow as tf
import keras.backend as K
import keras.layers as layers
import keras.initializers as kinit
import keras.regularizers as regularizers
import keras.constraints as constraints
import keras.models 
import utils.eval_funcs

if not os.path.isdir("../results/models"):
    os.mkdir("../results/models")
if not os.path.isdir("../results/models_tjs"):
    os.mkdir("../results/models_tjs")

costanzo_task_path = "../generated-data/task_yeast_gi_costanzo_thres16"
costanzo_targets_path = "../generated-data/targets/task_yeast_gi_costanzo_thres16_bin_interacting.npz"
costanzo_splits_path = "../generated-data/splits/task_yeast_gi_costanzo_thres16_10reps_4folds_0.20valid.npz"

def load_cfg(path, model_path, tjs_model_path, remove_specs=[], **kwargs):
    with open(path, 'r') as f:
        cfg = json.load(f)

    cfg['bootstrap_training'] = False 
    cfg['early_stopping'] = False 
    cfg['train_on_full_dataset'] = True
    cfg['train_model'] = True 
    cfg['epochs'] = 20
    cfg['trained_model_path'] = model_path
    cfg['save_tjs'] = True 
    cfg['tjs_path'] = tjs_model_path
    cfg['balanced_loss'] = True 

    cfg['spec'] = [s for s in cfg['spec'] if s['name'] not in remove_specs]
    cfg.update(kwargs)

    return cfg 

def generate_predictions(cfg, emodel, models, processors, thres, output_path, keep_net_preds=True):
    gpath = cfg['gpath']
    result_path = cfg['preds_path']
    main_df = pd.read_csv(cfg['task_path'])

    interacting_df = main_df[main_df['bin'] != 1]
    interacting_pairs = set('%d,%d' % tuple(sorted((a,b))) for a,b in zip(interacting_df['a_id'], interacting_df['b_id']))
    print("interacting pairs: %d" % len(interacting_pairs))
    
    noninteracting_df = main_df[main_df['bin'] == 1]
    noninteracting_pairs = set('%d,%d' % tuple(sorted((a,b))) for a,b in zip(noninteracting_df['a_id'], noninteracting_df['b_id']))
    print("noninteracting pairs: %d" % len(noninteracting_pairs))

    training_ids = set(main_df['a_id']).union(main_df['b_id'])

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())

    rows = []
    BATCH_SIZE = 250000
    stored = 0
    i = 0
    n_total = (len(nodes) * (len(nodes) - 1)) / 2

    first_append = True
    min_prob = 0.
    max_prob = 0.
    for comb in itertools.combinations(np.arange(len(nodes)), 2):
        rows.append({ 'a_id' : comb[0], 'b_id' : comb[1] })
        i += 1
        if len(rows) == BATCH_SIZE or i == n_total:
            df = pd.DataFrame(rows)
            df['pair'] = ['%d,%d' % t for t in zip(df['a_id'], df['b_id'])]
            in_interacting = df['pair'].isin(interacting_pairs)
            in_noninteracting = df['pair'].isin(noninteracting_pairs)
            df['observed'] = (in_interacting | in_noninteracting).astype(int)
            df['interacting'] =  in_interacting.astype(int)
            df['novel'] = (~df['a_id'].isin(training_ids) & ~df['b_id'].isin(training_ids)).astype(int)

            input_features = create_ensemble_features(models, df, processors)
            preds = emodel.predict(input_features, batch_size=cfg['batch_size'])

            min_prob = min(np.min(preds[:,0]), min_prob)
            max_prob = max(np.max(preds[:,0]), max_prob)
            preds_gi = preds[:,0] > thres

            if not keep_net_preds:
                ix = (preds_gi == 1) | df['observed']
            else:
                ix = (preds_gi == 1) | (preds_gi == 0)
            df['prob_gi'] = preds[:,0]
            
            output_df = df[ix][['a_id', 'b_id', 'prob_gi', 'observed', 'interacting', 'novel']]

            
            output_df.to_csv(output_path, mode='w' if first_append else 'a', header=first_append, index=False)
            first_append = False 

            stored += output_df.shape[0]
            print("Completed %8.4f, num interactions=%d" % (i / n_total, stored))

            rows = []

    print("Min: %0.3f, Max: %0.3f" % (min_prob, max_prob))

def examine_genes(cfg, result_path, gene_names, thresholds):

    gpath = cfg['gpath']
    ene_names = set(gene_names)

    df = pd.read_csv(result_path)
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    
    df['gene A'] = [nodes[a] for a in df['a_id']]
    df['gene B'] = [nodes[b] for b in df['b_id']]

    for gene in gene_names:
        print("Gene %s" % gene)


        ix = (df['gene A'] == gene) | (df['gene B'] == gene)

        print(" %d observations, %d interacting" % (np.sum((df[ix]['observed'] == 1)), np.sum(df[ix]['interacting'])))
        
        for t in thresholds:
            pred_gi = (df[ix]['prob_gi'] > t).astype(int)
            corr = (df[ix]['observed'] == 1) & (pred_gi == df[ix]['interacting'])
            mean_corr = np.sum(corr) / np.sum(df[ix]['observed'] == 1)
            novel_gi = pred_gi & (df[ix]['observed'] == 0)
            tpr = np.sum((df[ix]['observed'] == 1) & (pred_gi * df[ix]['interacting'] == 1)) / np.sum((df[ix]['observed'] == 1) & (df[ix]['interacting']==1))
            fpr = np.sum((df[ix]['observed'] == 1) & (df[ix]['interacting'] == 0) & (pred_gi == 1)) / np.sum((df[ix]['observed'] == 1) & (df[ix]['interacting']==0))
            print("  @ %0.2f, Accuracy: %0.2f, TPR: %0.2f, FPR: %0.2f, Novel GIs: %d" % (t, mean_corr, tpr, fpr, np.sum(novel_gi)))
        print() 

    print("Overall:")
    for t in thresholds:
        pred_gi = (df['prob_gi'] > t).astype(int)
        corr = (df['observed'] == 1) & (pred_gi == df['interacting'])
        mean_corr = np.sum(corr) / np.sum(df['observed'] == 1)
        novel_gi = pred_gi & (df['observed'] == 0)
        tpr = np.sum((df['observed'] == 1) & (pred_gi * df['interacting'] == 1)) / np.sum((df['observed'] == 1) & (df['interacting']==1))
        fpr = np.sum((df['observed'] == 1) & (df['interacting'] == 0) & (pred_gi == 1)) / np.sum((df['observed'] == 1) & (df['interacting']==0))
        print("  @ %0.2f, Accuracy: %0.2f, TPR: %0.2f, FPR: %0.2f, Novel GIs: %d" % (t, mean_corr, tpr, fpr, np.sum(novel_gi)))
    print() 

def create_training_inputs_outputs(mdl, cfg, other_cfgs):

    main_df = pd.read_csv(cfg['task_path'])

    cfg['train_model']= False
    model, processors = mdl.main(cfg, 0, 0, '../tmp/dummy', return_model=True )
    models = [model]
    for ocfg in other_cfgs:
        ocfg['train_model'] = False 
        om, _ = mdl.main(ocfg, 0, 0, '../tmp/dummy', return_model=True )
        models.append(om)

    inputs = create_ensemble_features(models, main_df, processors)
    outputs = np.zeros((main_df.shape[0], 2))
    outputs[main_df['bin'] != 1, 0] = 1
    outputs[main_df['bin'] == 1, 1] = 1
    
    return inputs, outputs, models, processors
    
def create_ensemble_features(models, df, processors, batch_size=10000):

    features = []
    for proc in processors:
        features.append(proc.transform(df))
            
    if len(processors) > 1:
        batch_F = np.hstack(features)
    else:
        batch_F = features 
    
    all_preds = []
    for model in models:
        # (n x n_classes)
        preds = model.predict(batch_F, batch_size=batch_size)
        preds = preds[:, 0]
        all_preds.append(preds)
    
    # (n_models x n)
    all_preds = np.array(all_preds)

    return all_preds.T
def weighted_categorical_xentropy(y_true, y_pred):
    
    xe = y_true * tf.math.log(y_pred)

    # (Kx1)
    xe = tf.reduce_sum(xe, axis=1, keepdims=True)

    # (1xC)
    class_freq = tf.reduce_sum(y_true, axis=0, keepdims=True)

    # (KxC) * (Cx1) = (Kx1)
    weights = tf.matmul(y_true, class_freq, transpose_b=True)
    
    wxe = -tf.reduce_sum(xe / weights)

    return wxe

mdl = models.gi_mn 

ycfg = load_cfg("cfgs/models/yeast_gi_mn.json",
    "../results/models/yeast_gi_mn_costanzo", 
    "../results/models_tjs/yeast_gi_mn_costanzo",
    splits_path=costanzo_splits_path,
    targets_path=costanzo_targets_path,
    task_path=costanzo_task_path)
ycfg['gpath'] = '../generated-data/ppc_yeast'
ycfg['preds_path'] = '../results/yeast_gi_preds'
#mdl.main(ycfg, 0, 0, '../tmp/dummy')

pcfg = load_cfg("cfgs/models/pombe_gi_mn.json",
    "../results/models/pombe_gi_mn", 
    "../results/models_tjs/pombe_gi_mn",
    targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz")
pcfg['gpath'] = '../generated-data/ppc_pombe'
pcfg['preds_path'] = '../results/pombe_gi_preds'
#mdl.main(pcfg, 0, 0, '../tmp/dummy')

hcfg = load_cfg("cfgs/models/human_gi_mn.json",
    "../results/models/human_gi_mn", 
    "../results/models_tjs/human_gi_mn")
hcfg['gpath'] = '../generated-data/ppc_human'
hcfg['preds_path'] = '../results/human_gi_preds'
#mdl.main(hcfg, 0, 0, '../tmp/dummy')

dcfg = load_cfg("cfgs/models/dro_gi_mn.json",
    "../results/models/dro_gi_mn", 
    "../results/models_tjs/dro_gi_mn")
dcfg['gpath'] = '../generated-data/ppc_dro'
dcfg['preds_path'] = '../results/dro_gi_preds'
#mdl.main(dcfg, 0, 0, '../tmp/dummy')

inputs, outputs, models, processors = create_training_inputs_outputs(mdl, ycfg, [pcfg, hcfg, dcfg])

input_node = layers.Input(shape=(inputs.shape[1],), name='input_features')
hidden_node = layers.Dense(5, activation='tanh')(input_node)
output_node = layers.Dense(outputs.shape[1], activation='softmax')(hidden_node)
loss = weighted_categorical_xentropy
emodel = keras.models.Model(inputs=input_node, outputs=output_node)
emodel.compile(ycfg['optimizer'], loss)

emodel.fit(inputs, outputs,
    shuffle=True,
    epochs=ycfg['epochs'],
    batch_size=ycfg['batch_size'],
    verbose=ycfg['verbose'])

preds = emodel.predict(inputs, batch_size=ycfg['batch_size'])
y_target = np.argmax(outputs, axis=1)

r, cm = utils.eval_funcs.eval_classifier(y_target, preds)
utils.eval_funcs.print_eval_classifier(r)

generate_predictions(ycfg, emodel, models, processors, 0.5, '../results/yeast_ensemble_preds')


examine_genes(ycfg, 
  '../results/yeast_ensemble_preds',
  ['ydr477w  snf1', 'yjr066w  tor1', 'ydl142c  crd1'], 
  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
