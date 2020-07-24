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

BIOGRID_PATH = '../generated-data/biogrid_yeast_gi'
COSTANZO_PATH = '../generated-data/costanzo_gi'

def main(gpath, output_path):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))

    biogrid_df = pd.read_csv(BIOGRID_PATH)
    biogrid_pairs = to_pairs(biogrid_df)
    costanzo_df = pd.read_csv(COSTANZO_PATH)

    # as per costanzo recommendation
    neg_ix = (costanzo_df['p_value'] < 0.05) & (costanzo_df['gi'] < -0.08)
    net_ix = (costanzo_df['p_value'] >= 0.05)
    pos_ix = (costanzo_df['p_value'] < 0.05) & (costanzo_df['gi'] > 0.08)
    neg_df = costanzo_df[neg_ix]
    net_df = costanzo_df[net_ix]
    pos_df = costanzo_df[pos_ix]
    neg_pairs = set([tuple(sorted(e)) for e in zip(neg_df['a'], neg_df['b'])])
    net_pairs = set([tuple(sorted(e)) for e in zip(net_df['a'], net_df['b'])])
    pos_pairs = set([tuple(sorted(e)) for e in zip(pos_df['a'], pos_df['b'])])
    pure_neg_pairs = neg_pairs - net_pairs - pos_pairs
    pure_pos_pairs = pos_pairs - net_pairs - neg_pairs
    pure_net_pairs = net_pairs - neg_pairs - pos_pairs

    costanzo_neutral_pairs = pure_net_pairs
    costanzo_neutral_pairs = costanzo_neutral_pairs - biogrid_pairs

    extra_rows = []

    for a, b in costanzo_neutral_pairs:
        extra_rows.append({
            "a" : a,
            "b" : b,
            "bin" : 1
        })
    
    df = biogrid_df.append(pd.DataFrame(extra_rows))

    ix = df['a'].isin(node_ix) & df['b'].isin(node_ix)
    df = df[ix]
    df['a_id'] = [node_ix[e] for e in df['a']]
    df['b_id'] = [node_ix[e] for e in df['b']]

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
    gpath = sys.argv[1]
    output_path = sys.argv[2]
    main(gpath, output_path)
    