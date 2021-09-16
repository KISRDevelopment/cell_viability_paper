import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import json
import os 
import sys

import utils.yeast_name_resolver 

res = utils.yeast_name_resolver.NameResolver()

def main(gpath, gaf_file, gene_name_col, output_path=None, all_go_terms=None, annotations_reader=None):
    
    G = nx.read_gpickle(gpath)
    nodes = list(sorted(G.nodes()))
    nodes_set = set(nodes)

    if not annotations_reader:
        annotations_reader = read_annotations
    genes_to_go = annotations_reader(gaf_file, gene_name_col)
    
    if not all_go_terms:
        all_go_terms = set()
        for k, associated_go in genes_to_go.items():
            if k.lower() in nodes_set:
                all_go_terms = all_go_terms.union(associated_go)
        all_go_terms = sorted(all_go_terms)
    
    all_go_terms = list(all_go_terms)
    
    terms_to_ids = dict(zip(all_go_terms, range(len(all_go_terms))))
    print("# of terms: %d" % len(all_go_terms))
    
    genes_without_go = len([1 for gene,gos in genes_to_go.items() if len(gos) == 0])
    print("# Genes without GO: %d" % genes_without_go)

    # create features 
    F = np.zeros((len(nodes), len(all_go_terms)))

    for i, node in enumerate(nodes):
        term_ids = [terms_to_ids[t] for t in genes_to_go[node]]
        F[i, term_ids] = 1

    if not output_path:
        output_path = '../generated-data/features/%s_sgo' % (os.path.basename(gpath))
        
    np.savez(output_path, F=F, feature_labels=all_go_terms)

    print(F.shape)
    print(np.sum(np.sum(F, axis=0) > 0))
    #print(all_go_terms)
    print(len(all_go_terms))

    n_terms_per_gene = np.sum(F, axis=1)
    print("Avg terms per gene: %f" % np.median(n_terms_per_gene))
def read_annotations(path, gene_name_col):

    genes_to_go = defaultdict(set)

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue

            parts = line.strip().split('\t')
        
            gene_name = parts[gene_name_col].lower()
            go_term = parts[4]

            genes_to_go[gene_name].add(go_term)
    
    return genes_to_go

def read_annotations_yeast(path, gene_name_col):

    genes_to_go = defaultdict(set)

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue

            parts = line.strip().split('\t')
        
            gene_name = res.get_unified_name(parts[10].lower().split('|')[0])
            go_term = parts[4]

            genes_to_go[gene_name].add(go_term)
    
    return genes_to_go

if __name__ == "__main__":
    gpath = sys.argv[1]
    gaf_file = sys.argv[2]
    gene_name_col = int(sys.argv[3])

    main(gpath, gaf_file, gene_name_col, annotations_reader=read_annotations_yeast)
    