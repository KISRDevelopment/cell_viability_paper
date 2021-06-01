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

def main(gpath):
    

    G = nx.read_gpickle(gpath)
    nodes = list(sorted(G.nodes()))
    nodes_set = set(nodes)

    df = pd.read_csv(GO_FILE, sep='\t', header=None, usecols=[0, 3, 4, 5], names=['name', 'aspect', 'term', 'id'])
    
    df_genes = list(df['name'])
    df_go = list(df['term'])
    df_aspect = list(df['aspect'])
    df_id = list(df['id'])

    # read all GO terms
    genes_to_go = { n: set() for n in nodes }
    aspects = set()
    for i in range(df.shape[0]):
        go = df_go[i]
        aspect = df_aspect[i]
        gene = df_genes[i]

        gene_name = res.get_unified_name(gene.lower())
        
        if gene_name not in nodes_set:
            continue
        
        term = '%s + %s' % (aspect, go)
        aspects.add(aspect)
        if term in to_remove:
            continue 
        
        genes_to_go[gene_name].add((aspect, go))

    print("Aspects")
    print(aspects)
    genes_without_go = len([1 for gene,gos in genes_to_go.items() if len(gos) == 0])
    print("# Genes without GO: %d" % genes_without_go)

    F = np.zeros((len(nodes), len(nodes), 3))
    for i in range(len(nodes)):
        terms_i = genes_to_go[nodes[i]]
        for j in range(i+1, len(nodes)):
            terms_j = genes_to_go[nodes[j]]

            #print("Shared P: %d, C: %d, F: %d" % (shared_p, shared_f, shared_c))
            F[i, j, :] = calculate_shared_terms(terms_i, terms_j)
            F[j, i, :] = F[i, j, :]
            
        print("Completed %d" % i)

    output_path = "../generated-data/pairwise_features/%s_shared_sgo" % (os.path.basename(gpath))
    
    np.save(output_path, F)

def calculate_shared_terms(terms_i, terms_j):

    shared_terms = terms_i.intersection(terms_j)

    shared_p = np.sum([1 for t in shared_terms if t[0] == 'P'])
    shared_f = np.sum([1 for t in shared_terms if t[0] == 'F'])
    shared_c = np.sum([1 for t in shared_terms if t[0] == 'C'])

    return shared_p, shared_f, shared_c

    # # create features 
    # F = np.zeros((len(nodes), 3))

    # for i, node in enumerate(nodes):
    #     term_ids = [terms_to_ids[t] for t in genes_to_go[node]]
    #     F[i, term_ids] = 1

    # output_path = '../generated-data/features/%s_sgo' % (os.path.basename(gpath))
    # all_go_terms = [names_to_goids[n] for n in all_go_terms]
    # np.savez(output_path, F=F, feature_labels=all_go_terms, names_to_goids=names_to_goids)

    # print(F.shape)
    
if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath)
    