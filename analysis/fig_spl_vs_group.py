import numpy as np 
import matplotlib.pyplot as plt 
import networkx as nx 
import sys 
import pandas as pd
import utils.yeast_name_resolver as nr
from collections import defaultdict
import json
import scipy.stats
res = nr.NameResolver()


def main(gpath, spl_path, group_type):

    if group_type == 'complex':
        genes_to_group = parse_yeast_complexes()
    else:
        genes_to_group = parse_kegg_pathways()
    
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))


    genes_to_group = {k:v for k,v in genes_to_group.items() if k in node_ix }
    

    F = np.load(spl_path)
    
    examined_genes = list(genes_to_group.keys())

    pair_spl = []
    pair_in_same_group = []
    for i in range(len(examined_genes)):
        group_i = genes_to_group[examined_genes[i]]
        idx_i = node_ix[examined_genes[i]]

        for j in range(i+1, len(examined_genes)):
            group_j = genes_to_group[examined_genes[j]]
            idx_j = node_ix[examined_genes[j]]

            pair_spl.append(F[idx_i, idx_j])
            pair_in_same_group.append(group_i == group_j)
    

    pair_spl = np.array(pair_spl)
    pair_in_same_group = np.array(pair_in_same_group).astype(bool)

    print("Average SPL for pairs in same group: %0.2f" % np.mean(pair_spl[pair_in_same_group]))
    print("Average SPL for pairs in different groups: %0.2f" % np.mean(pair_spl[~pair_in_same_group]))
    
    statistic, p = scipy.stats.ttest_ind(pair_spl[pair_in_same_group], pair_spl[~pair_in_same_group])
    print("T-test statistic: %f, p: %f" % (statistic,p))
def parse_kegg_pathways():

    with open('../data-sources/yeast/kegg_pathways', 'r') as f:
        genes_to_pathways = json.load(f)
    
    with open('../data-sources/yeast/kegg_names.json', 'r') as f:
        kegg_names = json.load(f)
    
    for k in genes_to_pathways.keys():
        pnames = [kegg_names[p] for p in genes_to_pathways[k]]
        genes_to_pathways[k] = pnames 

    genes_to_pathways = {res.get_unified_name(g) : set(genes_to_pathways[g]) for g in genes_to_pathways}

    return genes_to_pathways

def parse_yeast_complexes():

    df = pd.read_excel('../data-sources/yeast/CYC2008_complex.xls')

    df['gene'] = [res.get_unified_name(g.lower()) for g in df['ORF']]

    df_gene = list(df['gene'])
    df_complex = list(df['Complex'])

    genes_to_complexes = defaultdict(set)
    for i in range(df.shape[0]):
        g = df_gene[i]
        genes_to_complexes[g].add(df_complex[i])
    
    
    return {k:v for k,v in genes_to_complexes.items()}

if __name__ == "__main__":
    gpath = sys.argv[1]
    spl = sys.argv[2]
    group = sys.argv[3]

    main(gpath, spl, group)
