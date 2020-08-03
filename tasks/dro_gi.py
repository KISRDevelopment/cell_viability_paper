import os
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import sys
import networkx as nx
from collections import defaultdict
import numpy.random as rng 
import pickle
import json 
import itertools

BIOGRID_PATH = '../data-sources/dro/gene_genetic_interactions_fb_2020_01.tsv'

def main(gpath, output_path, n_samples=1000000):
    
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))
    
    df = pd.read_csv(BIOGRID_PATH, sep='\t', header=3)
    ix = ~pd.isnull(df['Starting_gene(s)_FBgn']) & ~pd.isnull(df['Interacting_gene(s)_FBgn'])
    df = df[ix]

    df_sys_a = list(df['Starting_gene(s)_FBgn'].astype(str).apply(get_fb))
    df_sys_b = list(df['Interacting_gene(s)_FBgn'].astype(str).apply(get_fb))
    df_condition = list(df['Interaction_type'])
    
    pair_conds = defaultdict(set)
    for i in range(df.shape[0]):

        a = df_sys_a[i].lower()
        b = df_sys_b[i].lower()
        cond = df_condition[i]

        if a in node_ix and b in node_ix:
            pair = tuple(sorted((a, b)))
            pair_conds[pair].add(cond)
    
    eligible_pairs = { k: list(v)[0] for k,v in pair_conds.items() if len(v) == 1 }
    
    rows = []
    for p, c in eligible_pairs.items():
        a, b = p 
        rows.append({
            "a" : a,
            "b" : b, 
            "a_id" : node_ix[a],
            "b_id" : node_ix[b],
            "bin" : 0 if c == 'enhanceable' else 2
        })

    rng.shuffle(nodes)
    neutral_pairs = set()
    for a, b in itertools.combinations(nodes, r=2):
        pair = tuple(sorted((a, b)))
        if pair in eligible_pairs:
            continue 
        
        neutral_pairs.add(pair)
        
        if len(neutral_pairs) == NEUTRAL_PAIRS:
            break
    
    for node_a, node_b in neutral_pairs:
        rows.append({
            "a" : node_a,
            "b" : node_b,
            "a_id" : node_ix[node_a],
            "b_id" : node_ix[node_b],
            "bin" : 1
        })


    df = pd.DataFrame(rows)

    bin = np.array(df['bin'])
    print([np.sum(bin == b) for b in [0,1,2,3]])

    df.to_csv(output_path, index=False)

def get_fb(s):
    parts = s.split('|')
    return parts[0]

if __name__ == "__main__":
    main()
    