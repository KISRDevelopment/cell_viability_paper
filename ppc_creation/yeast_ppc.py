import os
import sys
import pandas as pd
import networkx as nx
import numpy as np

from utils import yeast_name_resolver

thismodule = sys.modules[__name__]

coprespfiles = [
r'../data-sources/biogrid/BIOGRID-SYSTEM-Affinity_Capture-MS-3.4.156.mitab.txt',
r'../data-sources/biogrid/BIOGRID-SYSTEM-Affinity_Capture-Western-3.4.156.mitab.txt',
r'../data-sources/biogrid/BIOGRID-SYSTEM-Co-fractionation-3.4.156.mitab.txt',
r'../data-sources/biogrid/BIOGRID-SYSTEM-Co-purification-3.4.156.mitab.txt',
r'../data-sources/biogrid/BIOGRID-SYSTEM-Reconstituted_Complex-3.4.156.mitab.txt',
r'../data-sources/biogrid/BIOGRID-SYSTEM-Co-crystal_Structure-3.4.156.mitab.txt',
r'../data-sources/biogrid/BIOGRID-SYSTEM-Protein-peptide-3.4.156.mitab.txt',
]

# pseudo genes
genes_to_remove = ['yar062w  yar062w', 'yir044c  yir044c']

# translates common and locus tag names into one unified format
resolver = yeast_name_resolver.NameResolver()

# these are the genes that will be in the network
admissible_genes = resolver.get_genes()

# Set to False for GC only
ALL_GRAPH = False 

def main(output):
    
    copresp_G = nx.Graph()
    copresp_G.add_nodes_from(admissible_genes)
    for file_path in coprespfiles:
        print("Processing %s" % file_path)
        read_mitab_file(file_path, copresp_G)
    
    copresp_G.remove_nodes_from(genes_to_remove)
    print(nx.info(copresp_G))

    components = sorted(nx.connected_components(copresp_G), key=len, reverse=True)
    # write network to disk
    if not ALL_GRAPH:
        copresp_G = copresp_G.subgraph(components[0])
    print("Giant component net:")
    print(nx.info(copresp_G))
    
    
    # write network to disk
    nx.write_gpickle(copresp_G, output)
    nx.write_gml(copresp_G, output + '.gml')

def read_mitab_file(path, G):
    df = pd.read_csv(path, sep='\t')

    interactor_a = list(df['Alt IDs Interactor A'])
    interactor_b = list(df['Alt IDs Interactor B'])
    
    for (a, b) in zip(interactor_a, interactor_b):
        a = extract_locus_tag(a)
        b = extract_locus_tag(b)
        if a and b:
            G.add_edge(a,b)
    
def extract_locus_tag(s):
    part = s.split('|')[-1]
    part = s.split('/locuslink:')[-1].lower()

    # make sure name exists in the resolver's database of yeast genes
    if resolver.is_locus_name(part):
        un = resolver.get_unified_name(part)
        if un in admissible_genes:
            return un
    return None

if __name__ == "__main__":
    output = sys.argv[1]
    
    main(output)
    