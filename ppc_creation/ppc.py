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
        genes_to_remove = ['yar062w  yar062w', 'yir044c  yir044c']
        resolver = yeast_name_resolver.NameResolver()
        admissible_genes = resolver.get_genes()
        gc_only = True 
        extractor = lambda s: yeast_extract_locus_tag(s, resolver, admissible_genes)
        taxid = 559292
        
    elif organism == 'pombe':
        genes_to_remove = []
        gene_names = '../data-sources/pombe/PomBase2UniProt.csv'
        gene_names_df = pd.read_csv(gene_names, sep='\t', header=None, names=['locus', 'common'])
        admissible_genes =  set([str(e).lower() for e in gene_names_df['locus']])
        gc_only = False 
        extractor = lambda s: pombe_extract_locus_tag(s, admissible_genes)
        taxid = 284812

    elif organism == 'human':
        genes_to_remove = []
        gene_names = '../data-sources/human/gene_names'
        gene_names_df = pd.read_csv(gene_names, sep='\t')
        admissible_genes =  set([str(e).lower() for e in gene_names_df['Approved symbol']])
        gc_only = False 
        extractor = lambda s: pombe_extract_locus_tag(s, admissible_genes)
        taxid = 9606

    elif organism == "dro":
        genes_to_remove = []
        with open('../tmp/dro_gene_map.json', 'r') as f:
            en_fbgn = json.load(f)
        gc_only = False 
        extractor = lambda s: dro_extract_locus_tag(s, en_fbgn)
        admissible_genes = set([e.lower() for e in en_fbgn.values()])
        taxid = 7227
        reader = lambda file_path, copres_G, extractor, taxid: read_mitab_file(file_path, copres_G, extractor, taxid, '#ID Interactor A', 'ID Interactor B')
    
    copresp_G = nx.Graph()
    copresp_G.add_nodes_from(admissible_genes)
    for file_path in coprespfiles:
        print("Processing %s" % file_path)
        reader(file_path, copresp_G, extractor, taxid)
    
    copresp_G.remove_nodes_from(genes_to_remove)
    
    components = sorted(nx.connected_components(copresp_G), key=len, reverse=True)

    print("Full graph:")
    print(nx.info(copresp_G))
    
    # write network to disk
    if gc_only:
        copresp_G = copresp_G.subgraph(components[0])
    print(nx.info(copresp_G))
    
    # write network to disk
    nx.write_gpickle(copresp_G, output)
    nx.write_gml(copresp_G, output + '.gml')


def read_mitab_file(path, G, extractor, taxid, a_col='Alt IDs Interactor A', b_col='Alt IDs Interactor B'):
    df = pd.read_csv(path, sep='\t')

    interactor_a = list(df[a_col])
    interactor_b = list(df[b_col])
    df_org_a = list(df['Taxid Interactor A'])
    df_org_b = list(df['Taxid Interactor B'])

    taxid_str = "taxid:%d" % taxid 

    for (a, b, a_taxid, b_taxid) in zip(interactor_a, interactor_b, df_org_a, df_org_b):
        if a_taxid == taxid_str and b_taxid == taxid_str:
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
    