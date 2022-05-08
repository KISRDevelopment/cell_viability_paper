import os
from unicodedata import name
from nbformat import write
import pandas as pd
import sys
from collections import defaultdict
import utils.yeast_name_resolver
import json 
import networkx as nx 

thismodule = sys.modules[__name__]

BIOGRID_PATH = "../data-sources/biogrid/"
PATHS = [
    ("BIOGRID-SYSTEM-Synthetic_Lethality-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Synthetic_Growth_Defect-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Negative_Genetic-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Dosage_Growth_Defect-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Dosage_Lethality-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Phenotypic_Enhancement-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Phenotypic_Suppression-3.4.156.mitab", 3),
    ("BIOGRID-SYSTEM-Dosage_Rescue-3.4.156.mitab", 3),
    ("BIOGRID-SYSTEM-Synthetic_Rescue-3.4.156.mitab", 3),
    ("BIOGRID-SYSTEM-Positive_Genetic-3.4.156.mitab", 2)
]

FB_PATH = '../data-sources/dro/gene_genetic_interactions_fb_2020_01.tsv'
OUTPUT_PATH = 'website'

def main():
    os.makedirs(os.path.join(OUTPUT_PATH, 'data'), exist_ok=True)

    yeast_refs = extract_biogrid_refs(559292)
    pombe_refs = extract_biogrid_refs(284812)
    human_refs = extract_biogrid_refs(9606)
    dro_refs = extract_fb_refs()
    with open(os.path.join(OUTPUT_PATH, 'refs.json'), 'w') as f:
        json.dump({
            1 : yeast_refs,
            2 : pombe_refs,
            3 : human_refs,
            4 : dro_refs
        }, f)
        
    yeast_names = map_common_names_yeast()
    write_name_map(yeast_names, "yeast")

    pombe_names = map_common_names_pombe()
    write_name_map(pombe_names, "pombe")
    
    human_names = map_common_names_human()
    write_name_map(human_names, "human")
    
    dro_names = map_common_names_dro()
    write_name_map(dro_names, "dro")
    

def write_name_map(name_map, fname):
    with open(os.path.join(OUTPUT_PATH, 'data', '%s.json' % fname), 'w') as f:
        json.dump(name_map, f)

def extract_biogrid_refs(taxid):
    taxid_str = "taxid:%d" % taxid
    name_extraction_func = getattr(thismodule, "extract_names_taxid%d" % taxid)

    pairs_to_pubs = defaultdict(list)

    for path, condition in PATHS:
        df = pd.read_csv(os.path.join(BIOGRID_PATH, path+'.txt'), sep='\t')
    
        ix = (df['Taxid Interactor A'] == taxid_str) & (df['Taxid Interactor B'] == taxid_str)
        df = df[ix]

        a, b = name_extraction_func(df)

        publications = list(df['Publication Identifiers'])
        for i in range(df.shape[0]):
            key = tuple(sorted((a[i], b[i])))
            pairs_to_pubs[key].append(publications[i])

    return unzip(pairs_to_pubs)    
    
def unzip(d):
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    return keys, vals
def extract_names_taxid559292(df):

    res = utils.yeast_name_resolver.NameResolver()

    a_names = list(df['Alt IDs Interactor A'])
    b_names = list(df['Alt IDs Interactor B'])

    def extract_locus(e):
        return res.get_unified_name(e.split('|')[-1].replace('entrez gene/locuslink:', '').lower())

    return [extract_locus(e) for e in a_names], [extract_locus(e) for e in b_names]

def extract_names_taxid284812(df):

    a_names = list(df['Alt IDs Interactor A'])
    b_names = list(df['Alt IDs Interactor B'])

    def extract_locus(e):
        return e.split('|')[-1].replace('entrez gene/locuslink:', '').lower()

    return [extract_locus(e) for e in a_names], [extract_locus(e) for e in b_names]

extract_names_taxid9606 = extract_names_taxid284812

def extract_fb_refs():
    
    df = pd.read_csv(FB_PATH, sep='\t', header=3)
    ix = ~pd.isnull(df['Starting_gene(s)_FBgn']) & ~pd.isnull(df['Interacting_gene(s)_FBgn'])
    df = df[ix]

    df_sys_a = list(df['Starting_gene(s)_FBgn'].astype(str).apply(get_fb))
    df_sys_b = list(df['Interacting_gene(s)_FBgn'].astype(str).apply(get_fb))
    df_condition = list(df['Interaction_type'])
    df_pubs = list(df['Publication_FBrf'])

    pair_conds = defaultdict(set)
    pair_pubs = defaultdict(set)
    for i in range(df.shape[0]):
        a = df_sys_a[i].lower()
        b = df_sys_b[i].lower()
        cond = df_condition[i]
        pub = df_pubs[i]
        pair = tuple(sorted((a, b)))
        pair_conds[pair].add(cond)
        pair_pubs[pair].add(pub)


    # only allow pairs associated with one condition
    pairs_to_pubs = { k: list(pair_pubs[k]) for k,v in pair_conds.items() if len(v) == 1 }
    
    return unzip(pairs_to_pubs)

def get_fb(s):
    parts = s.split('|')
    return parts[0]

def map_common_names_yeast():

    G = nx.read_gpickle('../generated-data/ppc_yeast')
    
    full_names = sorted(G.nodes())

    tag_common = [n.split('  ') for n in full_names]
    tag_common = [b for a, b in tag_common]

    return { "locus" : full_names, "common" : tag_common }

def map_common_names_pombe():
    
    gene_names = '../data-sources/pombe/PomBase2UniProt.csv'
    gene_names_df = pd.read_csv(gene_names, sep='\t', header=None, names=['locus', 'common'])

    tags = gene_names_df['locus'].str.lower()
    common = gene_names_df['common'].fillna('').str.lower()

    tag_common = dict(zip(tags, common))

    G = nx.read_gpickle('../generated-data/ppc_pombe')
    full_names = sorted(G.nodes())
    #assert full_names.intersection(tags) == full_names 

    common = [tag_common[t] for t in full_names]

    return { "locus" : full_names, "common" : common }

def map_common_names_human():

    gene_names = '../data-sources/human/gene_names'
    gene_names_df = pd.read_csv(gene_names, sep='\t')

    tags = gene_names_df['Approved symbol'].fillna('').str.lower()
    common = gene_names_df['HGNC ID'].fillna('').str.lower()

    tag_common = dict(zip(tags, common))

    G = nx.read_gpickle('../generated-data/ppc_human')
    full_names = sorted(G.nodes())

    common = [tag_common.get(t, '') for t in full_names]

    return { "locus" : full_names, "common" : common }

def map_common_names_dro():

    MAP_FILE = "../data-sources/dro/fbgn_NAseq_Uniprot_fb_2020_01.tsv"

    df = pd.read_csv(MAP_FILE, sep='\t', header=4, na_filter=True)
    df = df[df['organism_abbreviation'] == 'Dmel']

    ix = ~pd.isnull(df['primary_FBgn#'])
    df = df[ix]

    tags = df['primary_FBgn#'].str.lower()
    common = df['gene_symbol'].fillna('').str.lower()

    tag_common = dict(zip(tags, common))

    G = nx.read_gpickle('../generated-data/ppc_dro')
    full_names = sorted(G.nodes())
    
    common = [tag_common[t] for t in full_names]

    return { "locus" : full_names, "common" : common }

if __name__ == "__main__":
    main()
