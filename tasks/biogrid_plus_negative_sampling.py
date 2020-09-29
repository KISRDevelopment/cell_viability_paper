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
import itertools 

def main(gpath, biogrid_path, smf_binned_path, output_path, n_samples=1000000, with_smf_only=True):
    
    biogrid_df = pd.read_csv(biogrid_path)
    biogrid_pairs = to_pairs(biogrid_df)

    G = nx.read_gpickle(gpath)
    nodes = np.array(sorted(G.nodes()))
    node_ix = dict(zip(nodes, np.arange(len(nodes))))
    
    if with_smf_only:
        d = np.load(smf_binned_path)
        F_smf = d['F']
        ix_has_smf = np.sum(F_smf, axis=1) > 0
        nodes = nodes[ix_has_smf]
        print("# nodes with SMF: %d" % len(nodes))

    N = len(nodes)
    combs = N*(N-1)/2
    print("# possible combs: %d" % combs)
    # add neutrals
    # anything that biogrid does not classify as interaction
    rng.shuffle(nodes)
    rows = []
    i = 0
    for a, b in itertools.combinations(nodes, r=2):
        if len(rows) == n_samples:
            break 

        pair = tuple(sorted((a, b)))
        if pair not in biogrid_pairs:
            rows.append({
                "a" : a, 
                "b" : b, 
                "bin" : 1 
            })

        i += 1
        if i % 100000 == 0:
            print("Finished %d (out of %d) Dataset size so far: %d" % (i, combs, len(rows)))
    df = biogrid_df.append(pd.DataFrame(rows))

    ix = df['a'].isin(node_ix) & df['b'].isin(node_ix)
    df = df[ix]
    df['a_id'] = [node_ix[e] for e in df['a']]
    df['b_id'] = [node_ix[e] for e in df['b']]

    ix = df['a_id'] != df['b_id']
    df = df[ix]

    print("Data size: ", df.shape)
    print("Bin counts:")
    print([np.sum(df['bin'] == b) for b in [0,1,2,3]])

    if with_smf_only:
        a_smf = F_smf[df['a_id'],:]
        b_smf = F_smf[df['b_id'],:]
        ix_a_no_smf = np.sum(a_smf, axis=1) == 0
        ix_b_no_smf = np.sum(b_smf, axis=1) == 0
        ix_no_smf_either = ix_a_no_smf | ix_b_no_smf
        df = df[~ix_no_smf_either]
        print("After filtering out pairs without SMF:")
        print("Data size: ", df.shape)
        print("Bin counts:")
        print([np.sum(df['bin'] == b) for b in [0,1,2,3]])

    df.to_csv(output_path, index=False)
    
def to_pairs(df):
    a = list(df['a'])
    b = list(df['b'])
    pairs = set()
    for i in range(df.shape[0]):
        pairs.add(tuple(sorted((a[i], b[i]))))
    return pairs

if __name__ == "__main__":
    main()
    