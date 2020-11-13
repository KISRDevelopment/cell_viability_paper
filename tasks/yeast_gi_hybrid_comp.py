import os
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import sys
import networkx as nx
from collections import defaultdict

COSTANZO_PATH = '../generated-data/costanzo_gi'
BIOGRID_PATH = '../generated-data/biogrid_yeast'

def main(gpath, output_path):
    
    
    costanzo_df = pd.read_csv(COSTANZO_PATH)

    biogrid_df = pd.read_csv(BIOGRID_PATH)
    biogrid_pairs = to_pairs(biogrid_df)

    # classify
    neg_ix = (costanzo_df['p_value'] < 0.05) & (costanzo_df['gi'] < -0.08)
    net_ix = (costanzo_df['p_value'] >= 0.05)
    pos_ix = (costanzo_df['p_value'] < 0.05) & (costanzo_df['gi'] > 0.08)
    gi_ix = neg_ix | pos_ix

    gi_df = costanzo_df[gi_ix]
    net_df = costanzo_df[net_ix]

    gi_pairs = set([tuple(sorted(e)) for e in zip(gi_df['a'], gi_df['b'])])
    net_pairs = set([tuple(sorted(e)) for e in zip(net_df['a'], net_df['b'])])

    # Neutral class is anything that doesn't occur in biogrid nor is classified as interacting
    gi_pairs = gi_pairs.union(biogrid_pairs)
    net_pairs = net_pairs - gi_pairs

    assert(len(gi_pairs.intersection(net_pairs)) == 0)

    rows = [
        { 
            "a" : a, 
            "b" : b, 
            "bin" : 0
        } for a,b in gi_pairs] + [
        {
            "a" : a,
            "b" : b, 
            "bin" : 1
        } for a, b in net_pairs]
    output_df = pd.DataFrame(rows)
    
    print("Data Size: %d, GI: %d, Neutral: %d" % (output_df.shape[0], len(gi_pairs), len(net_pairs)))

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))
    ix = output_df['a'].isin(node_ix) & output_df['b'].isin(node_ix)
    output_df = output_df[ix]
    print("After PPC Filter: Data Size: %d" % (output_df.shape[0]))

    output_df['a_id'] = [node_ix[e] for e in output_df['a']]
    output_df['b_id'] = [node_ix[e] for e in output_df['b']]


    output_df.to_csv(output_path, index=False)

    #output_df = pd.read_csv(output_path)
    coverage_by_gene = defaultdict(lambda: { "examinations" : 0, "interactions" : 0 })

    a_list = np.array(output_df['a'])
    b_list = np.array(output_df['b'])
    bin_list = np.array(output_df['bin'])
    for a, b, bin in zip(a_list, b_list, bin_list):

        coverage_by_gene[a]['examinations'] += 1
        coverage_by_gene[b]['examinations'] += 1
    
        if bin == 0:
            coverage_by_gene[a]['interactions'] += 1
            coverage_by_gene[b]['interactions'] += 1
    
    print("Covered genes: %d" % len(coverage_by_gene))

    exams_list = np.array([v['examinations'] for k, v in coverage_by_gene.items()])

    print("Median # examinations: %d" % np.median(exams_list))
    print("Range: %d - %d" % (np.min(exams_list), np.max(exams_list)))
    cutoffs = [0, 10, 100, 1000, 2000, 3000, 4000, 6000]
    for i in range(1, len(cutoffs)):
        start, end = cutoffs[i-1], cutoffs[i]
        print("  # genes examined %d-%d times: %d" % (start, end, np.sum((exams_list < end) & (exams_list >= start))))

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

    