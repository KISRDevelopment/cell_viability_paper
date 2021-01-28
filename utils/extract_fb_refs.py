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

def main(output_path = ""):
    
    df = pd.read_csv(BIOGRID_PATH, sep='\t', header=3)
    ix = ~pd.isnull(df['Starting_gene(s)_FBgn']) & ~pd.isnull(df['Interacting_gene(s)_FBgn'])
    df = df[ix]

    df_sys_a = list(df['Starting_gene(s)_FBgn'].astype(str).apply(get_fb))
    df_sys_b = list(df['Interacting_gene(s)_FBgn'].astype(str).apply(get_fb))
    df_condition = list(df['Interaction_type'])
    df_pubs = list(df['Publication_FBrf'])

    pair_conds = defaultdict(set)
    pair_pubs = defaultdict(set)
    for i in range(df.shape[0]):
        a = df_sys_a[i].lower()
        b = df_sys_b[i].lower()
        cond = df_condition[i]
        pub = df_pubs[i]
        pair = tuple(sorted((a, b)))
        pair_conds[pair].add(cond)
        pair_pubs[pair].add(pub)


    # only allow pairs associated with one condition
    pairs_to_pubs = { k: list(pair_pubs[k]) for k,v in pair_conds.items() if len(v) == 1 }
    
    if output_path:

        keys = list(pairs_to_pubs.keys())
        values = [pairs_to_pubs[k] for k in keys]

        output = {
            "pairs" : keys,
            "pubs" : values
        }
    
        with open(output_path, 'w') as f:
            json.dump(output, f)

    return pairs_to_pubs

def get_fb(s):
    parts = s.split('|')
    return parts[0]

if __name__ == "__main__":
    main(sys.argv[1])
    