#
# implementation of Lou 2015 IDC
#
import numpy as np 
import pandas as pd 
import networkx as nx 
import utils.yeast_name_resolver
from collections import defaultdict
import scipy.stats as stats 
import os 

res = utils.yeast_name_resolver.NameResolver()

def main(gpath):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    genes_to_complexes = parse_yeast_complexes()

    complexes_to_genes = defaultdict(set)
    for g, complexes in genes_to_complexes.items():
        for c in complexes:
            complexes_to_genes[c].add(g)

    idc = {}
    for g, complexes in genes_to_complexes.items():
        degrees = []
        for complex in complexes:
            subset = complexes_to_genes[complex]
            sG = G.subgraph(subset)
            degrees.append(sG.degree(g))
        idc[g] = np.sum(degrees)
    
    F = np.zeros((len(nodes), 1))
    for g, v in idc.items():
        if g in node_ix:
            F[node_ix[g], 0] = v 
    

    mu = np.mean(F, axis=0)
    std = np.std(F, axis=0)

    print(mu)
    print(std)

    # normalize
    F = stats.zscore(F, axis=0)
    
    print(np.min(F, axis=0))
    print(np.max(F, axis=0))
    print(np.mean(F, axis=0))
    print(np.std(F, axis=0))

    print("Nan: %d" % np.sum(np.isnan(F)))

    output_path = '../generated-data/features/%s_idc' % (os.path.basename(gpath))
    np.savez(output_path, F=F, feature_labels=['idc'], mu=mu, std=std)



def parse_yeast_complexes():

    df = pd.read_excel('../data-sources/yeast/CYC2008_complex.xls')

    df['gene'] = [res.get_unified_name(g.lower()) for g in df['ORF']]

    df_gene = list(df['gene'])
    df_complex = list(df['Complex'])

    genes_to_complexes = defaultdict(set)
    for i in range(df.shape[0]):
        g = df_gene[i]
        genes_to_complexes[g].add(df_complex[i])
    
    
    return genes_to_complexes

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
