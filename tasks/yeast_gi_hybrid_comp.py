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

def main(gpath, output_path, thres=0.08):
    
    
    costanzo_df = pd.read_csv(COSTANZO_PATH)

    biogrid_df = pd.read_csv(BIOGRID_PATH)
    biogrid_neg_pairs = to_pairs(biogrid_df[biogrid_df['bin'] == 0])
    biogrid_pos_pairs = to_pairs(biogrid_df[biogrid_df['bin'] == 2])
    biogrid_supp_pairs = to_pairs(biogrid_df[biogrid_df['bin'] == 3])
    
    # classify
    neg_ix = (costanzo_df['p_value'] < 0.05) & (costanzo_df['gi'] < -thres)
    net_ix = (costanzo_df['p_value'] >= 0.05)
    pos_ix = (costanzo_df['p_value'] < 0.05) & (costanzo_df['gi'] > thres)
    supp_ix = pos_ix & (costanzo_df['cs'] > np.maximum(costanzo_df['a_smf'], costanzo_df['b_smf']))

    neg_df = costanzo_df[neg_ix]
    net_df = costanzo_df[net_ix]
    pos_df = costanzo_df[pos_ix]
    supp_df = costanzo_df[supp_ix]

    neg_pairs = set([tuple(sorted(e)) for e in zip(neg_df['a'], neg_df['b'])])
    net_pairs = set([tuple(sorted(e)) for e in zip(net_df['a'], net_df['b'])])
    pos_pairs = set([tuple(sorted(e)) for e in zip(pos_df['a'], pos_df['b'])])
    supp_pairs = set([tuple(sorted(e)) for e in zip(supp_df['a'], supp_df['b'])])

    # combine with biogrid
    neg_pairs = neg_pairs.union(biogrid_neg_pairs)
    pos_pairs = pos_pairs.union(biogrid_pos_pairs)
    supp_pairs = supp_pairs.union(biogrid_supp_pairs)

    # purity filter: drop overlaps
    neg_pairs = neg_pairs - pos_pairs - supp_pairs
    pos_pairs = pos_pairs - neg_pairs - supp_pairs
    supp_pairs = supp_pairs - neg_pairs - pos_pairs

    # form neutral class
    net_pairs = net_pairs - neg_pairs - pos_pairs - supp_pairs

    # ensure no overlaps
    assert(len(neg_pairs.intersection(net_pairs)) == 0)
    assert(len(neg_pairs.intersection(pos_pairs)) == 0)
    assert(len(neg_pairs.intersection(supp_pairs)) == 0)
    assert(len(net_pairs.intersection(pos_pairs)) == 0)
    assert(len(net_pairs.intersection(supp_pairs)) == 0)
    assert(len(pos_pairs.intersection(supp_pairs)) == 0)

    pair_bins = [neg_pairs, net_pairs, pos_pairs, supp_pairs]
    rows = []
    for bin, pairs in enumerate(pair_bins):
        rows.extend([{
            "a" : a,
            "b" : b, 
            "bin" : bin 
        } for a, b in pairs])
    
    output_df = pd.DataFrame(rows)
    
    print("Data Size: %d" % (output_df.shape[0]))
    print([np.sum(output_df['bin'] == b) for b in [0,1,2,3]])

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))
    ix = output_df['a'].isin(node_ix) & output_df['b'].isin(node_ix)
    output_df = output_df[ix]
    print("After PPC Filter: Data Size: %d" % (output_df.shape[0]))
    print([np.sum(output_df['bin'] == b) for b in [0,1,2,3]])

    output_df['a_id'] = [node_ix[e] for e in output_df['a']]
    output_df['b_id'] = [node_ix[e] for e in output_df['b']]


    output_df.to_csv(output_path, index=False)

    # #output_df = pd.read_csv(output_path)
    coverage_by_gene = defaultdict(lambda: { "examinations" : 0, "interactions" : 0 })

    a_list = np.array(output_df['a'])
    b_list = np.array(output_df['b'])
    bin_list = np.array(output_df['bin'])
    for a, b, bin in zip(a_list, b_list, bin_list):

        coverage_by_gene[a]['examinations'] += 1
        coverage_by_gene[b]['examinations'] += 1
    
        if bin != 1:
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

    