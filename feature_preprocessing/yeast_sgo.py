import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import json
import os 
import sys

from utils import yeast_name_resolver

res = yeast_name_resolver.NameResolver()

GO_FILE = '../data-sources/yeast/go_slim_mapping.tab'

to_remove = ['C + other', 'P + other', 'F + other', 'F + not_yet_annotated', 'P + biological_process', 'C + cellular_component']
def main():
    gpath = sys.argv[1]

    G = nx.read_gpickle(gpath)
    nodes = list(sorted(G.nodes()))
    nodes_set = set(nodes)

    df = pd.read_csv(GO_FILE, sep='\t', header=None, usecols=[0, 3, 4, 5], names=['name', 'aspect', 'term', 'id'])
    
    df_genes = list(df['name'])
    df_go = list(df['term'])
    df_aspect = list(df['aspect'])
    df_id = list(df['id'])

    # read all GO terms
    all_go_terms = set()
    genes_to_go = { n: [] for n in nodes }
    names_to_goids = {}
    for i in range(df.shape[0]):
        go = df_go[i]
        aspect = df_aspect[i]
        gene = df_genes[i]

        gene_name = res.get_unified_name(gene.lower())
        
        if gene_name not in nodes_set:
            continue
        
        term = '%s + %s' % (aspect, go)

        if term in to_remove:
            continue 
        
        names_to_goids[term] = df_id[i]

        all_go_terms.add(term)
        genes_to_go[gene_name].append(term)

    all_go_terms = sorted(all_go_terms)
    terms_to_ids = dict(zip(all_go_terms, range(len(all_go_terms))))

    print("# of terms: %d" % len(all_go_terms))
    
    genes_without_go = len([1 for gene,gos in genes_to_go.items() if len(gos) == 0])
    print("# Genes without GO: %d" % genes_without_go)

    # create features 
    F = np.zeros((len(nodes), len(all_go_terms)))

    for i, node in enumerate(nodes):
        term_ids = [terms_to_ids[t] for t in genes_to_go[node]]
        F[i, term_ids] = 1

    output_path = '../generated-data/features/%s-sgo' % (os.path.basename(gpath))
    all_go_terms = [names_to_goids[n] for n in all_go_terms]
    np.savez(output_path, F=F, feature_labels=all_go_terms, names_to_goids=names_to_goids)

    print(F.shape)
    
if __name__ == "__main__":
    main()
    