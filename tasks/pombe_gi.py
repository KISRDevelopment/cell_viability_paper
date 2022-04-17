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

POMBE_GI = "../data-sources/pombe/Dataset S2 - Averaged E-MAP one allele per gene.txt"

def main(gpath, biogrid_path, smf_path, output_path):
    
    gi_df = read_source_gi()
    vals = np.array(gi_df)
    lower = np.nanpercentile(vals, 10)
    upper = np.nanpercentile(vals, 90)
    
    biogrid_df = pd.read_csv(biogrid_path)
    biogrid_pairs = to_pairs(biogrid_df)

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))
    
    # add neutrals
    # anything that biogrid does not classify as interaction
    row_indecies = [r for r in gi_df.index if r in node_ix]
    col_indecies = [c for c in gi_df.columns if c in node_ix]
    rows = []
    for r in row_indecies:
        for c in col_indecies:
            pair = tuple(sorted((r, c)))
            if pair not in biogrid_pairs and not np.isnan(gi_df.loc[r,c]):
                val = gi_df.loc[r, c]
                if val >= lower and val <= upper:
                    rows.append({
                        "a" : pair[0],
                        "b" : pair[1],
                        "bin" : 1
                    })
    
    df = biogrid_df.append(pd.DataFrame(rows))

    ix = df['a'].isin(node_ix) & df['b'].isin(node_ix)
    df = df[ix]
    df['a_id'] = [node_ix[e] for e in df['a']]
    df['b_id'] = [node_ix[e] for e in df['b']]

    print("Data size: ", df.shape)
    print("Bin counts:")
    print([np.sum(df['bin'] == b) for b in [0,1,2,3]])

    # filter out entries without SMF
    smf_df = pd.read_csv(smf_path)
    genes_with_smf = set(smf_df['id'])
    ix_no_smf_either = ~df['a_id'].isin(genes_with_smf) | ~df['b_id'].isin(genes_with_smf)
    df = df[~ix_no_smf_either]
    print("After filtering out pairs without SMF:")
    print("Data size: ", df.shape)
    print("Bin counts:")
    print([np.sum(df['bin'] == b) for b in [0,1,2,3]])

    df.to_csv(output_path, index=False)
    
def read_source_gi():


    gi_df = pd.read_csv(POMBE_GI, sep='\t')
    gi_df.columns = [c.lower() for c in gi_df.columns]
    gi_df['gene'] = gi_df['gene'].str.lower()

    gi_df = gi_df.set_index("gene")
    
    return gi_df

def to_pairs(df):
    a = list(df['a'])
    b = list(df['b'])
    pairs = set()
    for i in range(df.shape[0]):
        pairs.add(tuple(sorted((a[i], b[i]))))
    return pairs

if __name__ == "__main__":
    main()
    