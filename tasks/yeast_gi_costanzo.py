import os
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import sys
import networkx as nx
from collections import defaultdict

COSTANZO_PATH = '../generated-data/costanzo_gi'

def main(gpath, temps, allowed_combs, output_path):
    
    df = pd.read_csv(COSTANZO_PATH)

    # filter based on tempreatures
    ix = (df['temp'].isin(temps))
    df = df[ix]
    
    # filter based on allowed combinations
    e_combs = np.array(df[['a_essential', 'b_essential']])
    ix = None 
    for allowed_e_comb in allowed_combs:
        mix = np.prod(e_combs == allowed_e_comb, axis=1).astype(bool)
        if ix is None:
            ix = mix 
        ix = ix | mix
    df = df[ix]
    
    # only accept things in PPC
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    ix = df['a'].isin(node_ix) & df['b'].isin(node_ix)
    df = df[ix]

    df['a_id'] = [node_ix[a] for a in df['a']]
    df['b_id'] = [node_ix[b] for b in df['b']]
    print("Data size: ", df.shape)
    

    # as per costanzo recommendation
    neg_ix = (df['p_value'] < 0.05) & (df['gi'] < -0.08)
    net_ix = (df['p_value'] >= 0.05)
    pos_ix = (df['p_value'] < 0.05) & (df['gi'] > 0.08)

    neg_df = df[neg_ix]
    net_df = df[net_ix]
    pos_df = df[pos_ix]

    neg_pairs = set([tuple(sorted(e)) for e in zip(neg_df['a'], neg_df['b'])])
    net_pairs = set([tuple(sorted(e)) for e in zip(net_df['a'], net_df['b'])])
    pos_pairs = set([tuple(sorted(e)) for e in zip(pos_df['a'], pos_df['b'])])

    # ensure we have pairs that are exclusively reported in one of the three classes
    pure_neg_pairs = neg_pairs - net_pairs - pos_pairs
    pure_pos_pairs = pos_pairs - net_pairs - neg_pairs
    pure_net_pairs = net_pairs - neg_pairs - pos_pairs
    
    print("neg pairs: %d, neutral: %d, positive: %d" % (len(neg_pairs), len(net_pairs), len(pos_pairs)))
    print("Pure neg pairs: %d, neutral: %d, positive: %d" % (len(pure_neg_pairs), len(pure_net_pairs), len(pure_pos_pairs)))

    rows = []
    classify(neg_df, pure_neg_pairs, 0, rows)
    classify(pos_df, pure_pos_pairs, 2, rows)
    classify(net_df, pure_net_pairs, 1, rows)
        
    df = pd.DataFrame(rows)

    # add supression
    ix = (df['bin'] == 2) & (df['cs'] > np.maximum(df['a_cs'], df['b_cs']))
    bins = np.array(df['bin'])
    bins[ix] = 3
    df['bin'] = bins 

    print([np.sum(df['bin'] == b) for b in [0,1,2,3]])
    
    df.to_csv(output_path, index=False)

    print(df.describe())

def classify(df, pairs, bin, rows):
    df_a = list(df['a'])
    df_b = list(df['b'])
    df_a_id = list(df['a_id'])
    df_b_id = list(df['b_id'])
    df_gi = list(df['gi'])
    df_std = list(df['std'])
    df_cs = list(df['cs'])
    df_p_value = list(df['p_value'])
    df_a_smf = list(df['a_smf'])
    df_b_smf = list(df['b_smf'])
    df_a_essential = list(df['a_essential'])
    df_b_essential = list(df['b_essential'])
    df_temp = list(df['temp'])

    processed_pairs = set()
    for i in range(df.shape[0]):
        if i % 1000000 == 0:
            print(i)

        key = tuple(sorted((df_a[i], df_b[i])))
        if key in processed_pairs:
            continue 
        processed_pairs.add(key)

        if bin == 0:
            assert df_gi[i] < 0 
        elif bin == 2:
            assert df_gi[i] > 0
        
        if key in pairs:
            rows.append({
                "a" : df_a[i],
                "b" : df_b[i],
                "a_id" : df_a_id[i],
                "b_id" : df_b_id[i],
                "bin" : bin,
                "gi" : df_gi[i],
                "cs" : df_cs[i],
                "std" : df_std[i],
                "a_cs" : df_a_smf[i],
                "b_cs" : df_b_smf[i],
                "p_value" : df_p_value[i],
                "a_essential" : df_a_essential[i],
                "b_essential" : df_b_essential[i],
                "temp" : df_temp[i]
            })
if __name__ == "__main__":

    gpath = sys.argv[1]
    temps = [26] 
    allowed_combs = [(0, 0), (0, 1), (1, 0), (1, 1)] 
    output_path = sys.argv[2]

    main(gpath, temps, allowed_combs, output_path)

    