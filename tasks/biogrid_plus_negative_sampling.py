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

def main(gpath, biogrid_path, output_path, n_samples=1000000):
    
    biogrid_df = pd.read_csv(biogrid_path)
    biogrid_pairs = to_pairs(biogrid_df)

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))
    
    # add neutrals
    # anything that biogrid does not classify as interaction
    rng.shuffle(nodes)
    rows = []
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
    