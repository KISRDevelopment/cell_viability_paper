import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
import json
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

def main(organism, output):

    reader = read_mitab_file
    if organism == 'yeast':
        # pseudo genes
        genes_to_remove = ['yar062w  yar062w', 'yir044c  yir044c']
        # translates common and locus tag names into one unified format
        resolver = yeast_name_resolver.NameResolver()
        # these are the genes that will be in the network
        admissible_genes = resolver.get_genes()
        gc_only = True 
        extractor = lambda s: yeast_extract_locus_tag(s, resolver, admissible_genes)
        
    elif organism == 'pombe':
        genes_to_remove = []
        gene_names = '../data-sources/pombe/PomBase2UniProt.csv'
        gene_names_df = pd.read_csv(gene_names, sep='\t', header=None, names=['locus', 'common'])
        admissible_genes =  set([str(e).lower() for e in gene_names_df['locus']])
        gc_only = False 
        extractor = lambda s: pombe_extract_locus_tag(s, admissible_genes)
    
    elif organism == 'human':
        genes_to_remove = []
        gene_names = '../data-sources/human/gene_names'
        gene_names_df = pd.read_csv(gene_names, sep='\t')
        admissible_genes =  set([str(e).lower() for e in gene_names_df['Approved symbol']])
        gc_only = False 
        extractor = lambda s: pombe_extract_locus_tag(s, admissible_genes)
    
    elif organism == "dro":
        genes_to_remove = []
        with open('../tmp/dro_gene_map.json', 'r') as f:
            en_fbgn = json.load(f)
        gc_only = False 
        extractor = lambda s: dro_extract_locus_tag(s, en_fbgn)
        admissible_genes = set(list(en_fbgn.values()))
        reader = lambda file_path, copres_G, extractor: read_mitab_file(file_path, copres_G, extractor, '#ID Interactor A', 'ID Interactor B')
    
    copresp_G = nx.Graph()
    copresp_G.add_nodes_from(admissible_genes)
    for file_path in coprespfiles:
        print("Processing %s" % file_path)
        reader(file_path, copresp_G, extractor)
    
    copresp_G.remove_nodes_from(genes_to_remove)
    
    components = sorted(nx.connected_components(copresp_G), key=len, reverse=True)

    # write network to disk
    if gc_only:
        copresp_G = copresp_G.subgraph(components[0])
    print(nx.info(copresp_G))
    
    # write network to disk
    nx.write_gpickle(copresp_G, output)
    nx.write_gml(copresp_G, output + '.gml')


def read_mitab_file(path, G, extractor, a_col='Alt IDs Interactor A', b_col='Alt IDs Interactor B'):
    df = pd.read_csv(path, sep='\t')

    interactor_a = list(df[a_col])
    interactor_b = list(df[b_col])
    
    for (a, b) in zip(interactor_a, interactor_b):
        a = extractor(a)
        b = extractor(b)
        if a and b:
            G.add_edge(a,b)
    
def yeast_extract_locus_tag(s, resolver, admissible_genes):
    part = s.split('|')[-1]
    part = s.split('/locuslink:')[-1].lower()

    # make sure name exists in the resolver's database of yeast genes
    if resolver.is_locus_name(part):
        un = resolver.get_unified_name(part)
        if un in admissible_genes:
            return un
    return None

def pombe_extract_locus_tag(s, admissible_genes):
    part = s.split('|')[-1]
    part = s.split('/locuslink:')[-1].lower()

    if part in admissible_genes:
        return part 
    
    return None

def dro_extract_locus_tag(s, en_fbgn):
    s = s.lower().replace('entrez gene/locuslink:','')
    if s in en_fbgn:
        return en_fbgn[s].lower()
    return None 

if __name__ == "__main__":
    organism = sys.argv[1]
    output = sys.argv[2]
    
    main(organism, output)
    