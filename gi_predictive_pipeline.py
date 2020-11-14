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

if not os.path.isdir("../results/models"):
    os.mkdir("../results/models")
if not os.path.isdir("../results/models_tjs"):
    os.mkdir("../results/models_tjs")

costanzo_task_path = "../generated-data/task_yeast_gi_costanzo"
costanzo_targets_path = "../generated-data/targets/task_yeast_gi_costanzo_bin_interacting.npz"
costanzo_splits_path = "../generated-data/splits/task_yeast_gi_costanzo_10reps_4folds_0.20valid.npz"

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

mdl = models.gi_mn 

#
# Train Species Models
#
yeast_cfg = load_cfg("cfgs/models/yeast_gi_mn.json",
    "../results/models/yeast_gi_mn", 
    "../results/models_tjs/yeast_gi_mn",
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz")
yeast_cfg_nosmf = load_cfg("cfgs/models/yeast_gi_mn.json",
    "../results/models/yeast_gi_mn_nosmf", 
    "../results/models_tjs/yeast_gi_mn_nosmf",
    remove_specs=['smf'],
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz")
yeast_cfg_nosgo = load_cfg("cfgs/models/yeast_gi_mn.json",
    "../results/models/yeast_gi_mn_nosmf", 
    "../results/models_tjs/yeast_gi_mn_nosmf",
    remove_specs=['sgo'],
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz")
yeast_cfg_refined= load_cfg("cfgs/models/yeast_gi_refined_model.json",
    "../results/models/yeast_gi_refined", 
    "../results/models_tjs/yeast_gi_refined",
    targets_path="../generated-data/targets/task_yeast_gi_hybrid_bin_interacting.npz")
yeast_cfg_costanzo = load_cfg("cfgs/models/yeast_gi_mn.json",
    "../results/models/yeast_gi_mn_costanzo", 
    "../results/models_tjs/yeast_gi_mn_costanzo",
    splits_path=costanzo_splits_path,
    targets_path=costanzo_targets_path,
    task_path=costanzo_task_path)

costanzo_task_path = "../generated-data/task_yeast_gi"
costanzo_targets_path = "../generated-data/targets/task_yeast_gi_bin_interacting.npz"
costanzo_splits_path = "../generated-data/splits/task_yeast_gi_10reps_4folds_0.20valid.npz"
yeast_cfg_def = load_cfg("cfgs/models/yeast_gi_mn.json",
    "../results/models/yeast_gi_mn_def", 
    "../results/models_tjs/yeast_gi_mn_def",
    splits_path=costanzo_splits_path,
    targets_path=costanzo_targets_path,
    task_path=costanzo_task_path)

yeast_cfg_costanzo_nosmf = load_cfg("cfgs/models/yeast_gi_mn.json",
    "../results/models/yeast_gi_mn_costanzo_nosmf", 
    "../results/models_tjs/yeast_gi_mn_costanzo_nosmf",
    splits_path=costanzo_splits_path,
    targets_path=costanzo_targets_path,
    task_path=costanzo_task_path,
    remove_specs=['smf'])
pombe_cfg = load_cfg("cfgs/models/pombe_gi_mn.json",
    "../results/models/pombe_gi_mn", 
    "../results/models_tjs/pombe_gi_mn",
    targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz")

human_cfg = load_cfg("cfgs/models/human_gi_mn.json",
    "../results/models/human_gi_mn", 
    "../results/models_tjs/human_gi_mn")

dro_cfg = load_cfg("cfgs/models/dro_gi_mn.json",
    "../results/models/dro_gi_mn", 
    "../results/models_tjs/dro_gi_mn",
    remove_specs=["sgo", "smf", "topology"])


#mdl.main(yeast_cfg, 0, 0, '../tmp/dummy')
#mdl.main(yeast_cfg_nosmf, 0, 0, '../tmp/dummy')
#mdl.main(yeast_cfg_nosgo, 0, 0, '../tmp/dummy')
#mdl.main(yeast_cfg_costanzo_nosmf, 0, 0, '../tmp/dummy')
#mdl.main(yeast_cfg_def, 0, 0, '../tmp/dummy')
#models.gi_nn.main(yeast_cfg_refined, 0, 0, '../tmp/dummy')

# mdl.main(pombe_cfg, 0, 0, '../tmp/dummy')
# mdl.main(human_cfg, 0, 0, '../tmp/dummy')
#print(dro_cfg)
#mdl.main(dro_cfg, 0, 0, '../tmp/dummy')

def generate_predictions(mdl, cfg, gpath, result_path, thres):

    main_df = pd.read_csv(cfg['task_path'])

    interacting_df = main_df[main_df['bin'] != 1]
    interacting_pairs = set('%d,%d' % tuple(sorted((a,b))) for a,b in zip(interacting_df['a_id'], interacting_df['b_id']))
    print("interacting pairs: %d" % len(interacting_pairs))
    
    noninteracting_df = main_df[main_df['bin'] == 1]
    noninteracting_pairs = set('%d,%d' % tuple(sorted((a,b))) for a,b in zip(noninteracting_df['a_id'], noninteracting_df['b_id']))
    print("noninteracting pairs: %d" % len(noninteracting_pairs))

    training_ids = set(main_df['a_id']).union(main_df['b_id'])

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

            rows = []

            features = []
            for proc in processors:
                features.append(proc.transform(df))
            
            if len(processors) > 1:
                batch_F = np.hstack(features)
            else:
                batch_F = features 
            
            preds = model.predict(batch_F, batch_size=BATCH_SIZE)
            min_prob = min(np.min(preds[:,0]), min_prob)
            max_prob = max(np.max(preds[:,0]), max_prob)
            preds_gi = preds[:,0] > thres

            #ix = (preds_gi == 1) | df['observed']
            ix = (preds_gi == 1) | (preds_gi == 0)
            df['prob_gi'] = preds[:,0]
            
            output_df = df[ix][['a_id', 'b_id', 'prob_gi', 'observed', 'interacting', 'novel']]

            
            output_df.to_csv(result_path, mode='w' if first_append else 'a', header=first_append, index=False)
            first_append = False 

            stored += output_df.shape[0]
            print("Completed %8.4f, num interactions=%d" % (i / n_total, stored))

    print("Min: %0.3f, Max: %0.3f" % (min_prob, max_prob))

#generate_predictions(mdl, yeast_cfg, '../generated-data/ppc_yeast', '../results/yeast_gi_preds', 0.5)
#generate_predictions(yeast_cfg_nosmf, '../generated-data/ppc_yeast', '../results/yeast_gi_preds_nosmf', 0.5)
#generate_predictions(models.gi_mn, yeast_cfg_nosgo, '../generated-data/ppc_yeast', '../results/yeast_gi_preds_nosgo', 0.5)
#generate_predictions(models.gi_mn, yeast_cfg_costanzo, '../generated-data/ppc_yeast', '../results/yeast_gi_preds_costanzo', 0.5)
#generate_predictions(models.gi_mn, yeast_cfg_costanzo_nosmf, '../generated-data/ppc_yeast', '../results/yeast_gi_preds_costanzo_nosmf', 0.5)
#generate_predictions(models.gi_mn, yeast_cfg_def, '../generated-data/ppc_yeast', '../results/yeast_gi_preds_def', 0.5)

#generate_predictions(models.gi_nn, yeast_cfg_refined, '../generated-data/ppc_yeast', '../results/yeast_gi_preds_refined', 0.5)

#generate_predictions(pombe_cfg, '../generated-data/ppc_pombe', '../results/pombe_gi_preds')
#generate_predictions(dro_cfg, '../generated-data/ppc_dro', '../results/dro_gi_preds')
#generate_predictions(human_cfg, '../generated-data/ppc_human', '../results/human_gi_preds')

def select_subset(gpath, result_path, gene_names, output_path, thres=0.5):
    gene_names = set(gene_names)

    df = pd.read_csv(result_path)
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    
    df['gene A'] = [nodes[a] for a in df['a_id']]
    df['gene B'] = [nodes[b] for b in df['b_id']]

    ix = df['gene A'].isin(gene_names) | df['gene B'].isin(gene_names)

    df = df[ix]

    df['prediction'] = (df['prob_gi'] > thres).astype(int) 
    df['correct'] = (df['observed'] == 1) & (df['prediction'] == df['interacting'])
    df.to_excel(output_path + '.xlsx', columns=['gene A', 'gene B', 'prob_gi', 'observed', 'interacting', 'prediction', 'correct'], index=False)

def examine_genes(gpath, result_path, gene_names, thresholds):

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


examine_genes('../generated-data/ppc_yeast', '../results/yeast_gi_preds_def', 
   ['ydr477w  snf1', 'yjr066w  tor1', 'ydl142c  crd1'], 
   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# #select_subset('../generated-data/ppc_human', '../results/human_gi_preds', ['myc', 'tp53'], '../results/human_gi_preds_subset')
# #select_subset('../generated-data/ppc_dro', '../results/dro_gi_preds', ['fbgn0003366', 'fbgn0024248'], '../results/dro_gi_preds_subset')
# #select_subset('../generated-data/ppc_dro', '../results/dro_gi_preds', ['fbgn0003366'], '../results/dro_gi_preds_fbgn0003366')
#select_subset('../generated-data/ppc_yeast', '../results/yeast_gi_preds', ['ydr477w  snf1'], '../results/yeast_gi_preds_snf1', 0.5)
#select_subset('../generated-data/ppc_yeast', '../results/yeast_gi_preds_nosmf', ['yjr066w  tor1'], '../results/yeast_gi_preds_tor1', 0.7)
# select_subset('../generated-data/ppc_yeast', '../results/yeast_gi_preds', ['ykl203c  tor2'], '../results/yeast_gi_preds_tor2')
# # select_subset('../generated-data/ppc_yeast', '../results/yeast_gi_preds', ['ypl178w  cbc2'], '../results/yeast_gi_preds_cbc2')
# #select_subset('../generated-data/ppc_yeast', '../results/yeast_gi_preds', ['yil096c  bmt5'], '../results/yeast_gi_preds_cbc2')
# #select_subset('../generated-data/ppc_yeast', '../results/yeast_gi_preds', ['ydr001c  nth1'], '../results/yeast_gi_preds_cbc2')
# #select_subset('../generated-data/ppc_yeast', '../results/yeast_gi_preds', ['ydl037c  bsc1'], '../results/yeast_gi_preds_cbc2')
# #select_subset('../generated-data/ppc_yeast', '../results/yeast_gi_preds', ['ylr099c  ict1'], '../results/yeast_gi_preds_cbc2')
# select_subset('../generated-data/ppc_yeast', '../results/yeast_gi_preds', ['ydl142c  crd1'], '../results/yeast_gi_preds_cbc2')

def sanity_check(result_path):

    df = pd.read_csv(result_path)

    df_obs = df[df['observed'] == 1]

    
    ypred = df_obs['prob_gi'] > 0.5
    ytarget = df_obs['interacting']


    cm = sklearn.metrics.confusion_matrix(ytarget, ypred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    print(cm)

    import matplotlib.pyplot as plt 

    fig, axes = plt.subplots(1, 2, figsize=(20,20))
    axes[0].hist(df_obs['prob_gi'], bins=20, density=True)
    axes[0].set_title('On Reported Data')
    axes[1].hist(df[df['observed']==0]['prob_gi'], bins=20, density=True)
    axes[1].set_title('On Unreported Data')
    #axes[1,0].hist(df[df['novel']==1]['prob_gi'], bins=20, density=True)
    #axes[1,1].hist(df[(df['novel']==0)&(df['observed']==0)]['prob_gi'], bins=20, density=True)
    plt.show()

#sanity_check('../results/yeast_gi_preds_costanzo_nosmf')

# d = np.load('../results/task_yeast_gi_hybrid_binary/mn/run_6_2.npz', allow_pickle=True)
# #d = np.load('../tmp/dummy.npz', allow_pickle=True)

# y_target = d['y_target']

# y_pred = d['preds'][:,0]

# import matplotlib.pyplot as plt 

# f, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.hist(y_pred, bins=20)
# plt.show()

def spl_check(result_path, spl_path):

    df = pd.read_csv(result_path)

    F = np.load(spl_path)

    df['spl'] = F[df['a_id'], df['b_id']]

    df_obs = df[df['observed']==1]
    df_nonobs = df[df['observed']==0]

    import matplotlib.pyplot as plt 

    f, axes = plt.subplots(1, 2, figsize=(20,10))
    axes[0].hist(df_obs['spl'], bins=20)
    axes[1].hist(df_nonobs['spl'], bins=20)
    plt.show()

#spl_check('../results/yeast_gi_preds', '../generated-data/pairwise_features/ppc_yeast_shortest_path_len.npy')

def sum_lid_check(result_path, lid_path):

    df = pd.read_csv(result_path)

    d = np.load(lid_path)
    ix = d['feature_labels'] == 'lid'
    F = d['F'][:, ix]

    df['sum_lid'] = F[df['a_id']] + F[df['b_id']]

    df_obs = df[df['observed']==1]
    df_nonobs = df[df['observed']==0]

    import matplotlib.pyplot as plt 

    f, axes = plt.subplots(1, 2, figsize=(20,10))
    axes[0].hist(df_obs['sum_lid'], bins=20)
    axes[1].hist(df_nonobs['sum_lid'], bins=20)
    plt.show()
#sum_lid_check('../results/yeast_gi_preds', '../generated-data/features/ppc_yeast_topology.npz')

def smf_check(result_path, path):

    df = pd.read_csv(result_path)

    d = np.load(path)
    F = d['F']

    smf_features = []
    for u in range(F.shape[1]):
        for w in range(u, F.shape[1]):
            smf_features.append(
                (F[df['a_id'], u] * F[df['b_id'], w]).astype(bool) | \
                (F[df['a_id'], w] * F[df['b_id'], u]).astype(bool))
                
    smf_features = np.vstack(smf_features).T.astype(int)

    ix_obs = df['observed'] == 1
    ix_nonobs = ~ix_obs 

    obs_counts = np.sum(smf_features[ix_obs, :], axis=0)
    nonobs_counts = np.sum(smf_features[ix_nonobs, :], axis=0)

    import matplotlib.pyplot as plt 

    f, axes = plt.subplots(2, 1, figsize=(20, 10), sharey=True)

    vals = ['LL', 'LR', 'LN', 'RR', 'RN', 'NN']
    axes[0].bar(vals, obs_counts / np.sum(obs_counts))
    axes[0].set_title('Reported')

    axes[1].bar(vals, nonobs_counts / np.sum(nonobs_counts))
    axes[1].set_title('Unreported')

    plt.show()

#smf_check('../results/yeast_gi_preds_costanzo', '../generated-data/features/ppc_yeast_smf_binned.npz')

