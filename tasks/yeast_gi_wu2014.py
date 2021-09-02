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
import utils.yeast_name_resolver

res = utils.yeast_name_resolver.NameResolver()

ESSENTIALS_PATH = "../data-sources/wu2014/essential-genes.txt"

def main(gpath, biogrid_path, output_path):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))

    biogrid_df = pd.read_csv(biogrid_path)
    print(biogrid_df.columns)
    biogrid_df = biogrid_df[biogrid_df['bin'] == 0]
    biogrid_pairs = to_pairs(biogrid_df)
    biogrid_pairs = set([p for p in biogrid_pairs if (p[0] in node_ix) and (p[1] in node_ix)])
    print("SL Pairs: %d" % len(biogrid_pairs))
    
    essentials = get_our_essentials()
    print("Read %d essentials" % len(essentials))

    # exclude pairs including essentials
    biogrid_pairs = set([p for p in biogrid_pairs if (p[0] not in essentials) and (p[1] not in essentials)])
    print("SL pairs w/o essentials: %d" % len(biogrid_pairs))

    # randomly sample non-essential pairs
    nonessentials = list(set(node_ix.keys()) - essentials)
    print("Length nonessentials: %d" % len(nonessentials))

    candidates_a = rng.permutation(30000) % len(nonessentials)
    candidates_b = rng.permutation(30000) % len(nonessentials)

    candidate_pairs = set([tuple(sorted((nonessentials[a],nonessentials[b]))) for a,b in zip(candidates_a, candidates_b) if a != b])
    print("Length candidates: %d" % len(candidate_pairs))

    candidate_pairs = candidate_pairs - biogrid_pairs
    print("Length candidates without biogrid pairs: %d" % len(candidate_pairs))

    non_sl_pairs = list(candidate_pairs)[:len(biogrid_pairs)]

    rows = [ { "a" : p[0], "b" : p[1], "bin" : 0, "a_id" : node_ix[p[0]], "b_id" : node_ix[p[1]] } for p in biogrid_pairs] + \
           [ { "a" : p[0], "b" : p[1], "bin" : 1, "a_id" : node_ix[p[0]], "b_id" : node_ix[p[1]] } for p in non_sl_pairs]
    output_df = pd.DataFrame(rows)

    print([np.sum(output_df['bin'] == b) for b in [0,1]])

    output_df.to_csv(output_path, index=False)
    
def get_essentials():

    essentials = set()
    with open(ESSENTIALS_PATH, 'r') as f:
        for line in f:
            common, locus = re.split(r'\s+', line.strip())
            essentials.add(res.get_unified_name(locus))

    return essentials

def get_our_essentials():

    df = pd.read_csv("../generated-data/task_yeast_smf_30")

    return set(df[df['bin'] == 0]['gene'])

def to_pairs(df):
    a = list(df['a'])
    b = list(df['b'])
    pairs = set()
    for i in range(df.shape[0]):
        pairs.add(tuple(sorted((a[i], b[i]))))
    return pairs

if __name__ == "__main__":
    gpath = sys.argv[1]
    biogrid_path = sys.argv[2]
    output_path = sys.argv[3]

    main(gpath, biogrid_path, output_path)
    