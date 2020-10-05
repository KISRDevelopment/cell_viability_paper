import pandas as pd 
import numpy as np 
import sys
import utils.yeast_name_resolver as nr
from collections import defaultdict 
import json
import scipy.stats
import numpy.random as rng
import networkx as nx
res = nr.NameResolver()

def main(gpath, task_path, smf_path, output_path):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    gi_df = pd.read_csv(task_path)
    d = np.load(smf_path)
    Fsmf = d['F']

    a_smf = np.argmax(Fsmf[gi_df['a_id'], :], axis=1)
    b_smf = np.argmax(Fsmf[gi_df['b_id'], :], axis=1)
    a_has_smf_ix = np.sum(Fsmf[gi_df['a_id'], :], axis=1) > 0
    b_has_smf_ix = np.sum(Fsmf[gi_df['b_id'], :], axis=1) > 0
    both_have_smf_ix = a_has_smf_ix & b_has_smf_ix
    gi_df = gi_df[both_have_smf_ix]

    print("Complexes:")
    genes_to_complexes = parse_yeast_complexes()
    genes_to_complexes, complexes = filter_membership(genes_to_complexes, node_ix)
    complex_ix, same_complex_ix = make_group(genes_to_complexes, gi_df, Fsmf)

    print("Pathways:")
    genes_to_pathways = parse_kegg_pathways()
    genes_to_pathways, pathways = filter_membership(genes_to_pathways, node_ix)
    pathway_ix, same_pathway_ix = make_group(genes_to_pathways, gi_df, Fsmf)

    diff_complex_same_pathway_ix = complex_ix & ~same_complex_ix & same_pathway_ix
    print("Different complex but same pathway: %d" % np.sum(diff_complex_same_pathway_ix))

    complex_df = gi_df[complex_ix & ~diff_complex_same_pathway_ix].copy()
    complex_df['a_group'] = [genes_to_complexes[g] for g in complex_df['a_id']]
    complex_df['b_group'] = [genes_to_complexes[g] for g in complex_df['b_id']]
    complex_df['same_group'] = (complex_df['a_group'] == complex_df['b_group']).astype(int)
    complex_df['a_smf'] = np.argmax(Fsmf[complex_df['a_id'], :], axis=1)
    complex_df['b_smf'] = np.argmax(Fsmf[complex_df['b_id'], :], axis=1)
    print(complex_df.shape)

    
    same_pathway_complex_ix = pathway_ix & same_complex_ix & same_pathway_ix
    print("Same complex and pathway: %d" % np.sum(same_pathway_complex_ix))
    pathway_df = gi_df[pathway_ix & ~same_pathway_complex_ix].copy()
    pathway_df['a_group'] = [genes_to_pathways[g] for g in pathway_df['a_id']]
    pathway_df['b_group'] = [genes_to_pathways[g] for g in pathway_df['b_id']]
    pathway_df['same_group'] = (pathway_df['a_group'] == pathway_df['b_group']).astype(int)
    pathway_df['a_smf'] = np.argmax(Fsmf[pathway_df['a_id'], :], axis=1)
    pathway_df['b_smf'] = np.argmax(Fsmf[pathway_df['b_id'], :], axis=1)
    print(pathway_df.shape)

    complex_sizes = compute_group_size(genes_to_complexes)
    pd.DataFrame({
        "group" : complexes,
        "n_genes" : [complex_sizes[c] for c in range(len(complexes))],
        "n_genes_glob" : len(genes_to_complexes),
    }).to_csv(output_path + '_complexes', index=False)

    pathways_sizes = compute_group_size(genes_to_pathways)
    pd.DataFrame({
        "group" : pathways,
        "n_genes" : [pathways_sizes[c] for c in range(len(pathways))],
        "n_genes_glob" : len(genes_to_pathways),
    }).to_csv(output_path + '_pathways', index=False)

    complex_df.to_csv(output_path + '_complex_pairs', 
        columns=['a_id', 'b_id', 'bin', 'a_group', 'b_group', 'same_group', 'a_smf', 'b_smf'], 
        index=False)
    pathway_df.to_csv(output_path + '_pathway_pairs', 
        columns=['a_id', 'b_id', 'bin', 'a_group', 'b_group', 'same_group', 'a_smf', 'b_smf'], 
        index=False)
    
def compute_group_size(genes_to_group):
    group_sizes = defaultdict(lambda: 0)
    for _, g in genes_to_group.items():
        group_sizes[g] += 1
    return group_sizes

def make_group(genes_to_group, gi_df, Fsmf):

    a_group = np.array([genes_to_group.get(g, None) for g in gi_df['a_id']])
    b_group = np.array([genes_to_group.get(g, None) for g in gi_df['b_id']])
    a_has_group_ix = np.array([g is not None for g in a_group])
    b_has_group_ix = np.array([g is not None for g in b_group])
    both_have_group_ix = a_has_group_ix & b_has_group_ix
    both_have_same_group = both_have_group_ix & (a_group == b_group)
    print("# entries with group: %d, same group: %d" % (np.sum(both_have_group_ix), np.sum(both_have_same_group)))

    return both_have_group_ix, both_have_same_group

def filter_membership(genes_to_group, node_ix):
    
    # limit to copresp
    genes_to_group = { g:v for g,v in genes_to_group.items() if g in node_ix}

    # one group assignment
    genes_to_group = { node_ix[g]: list(v)[0] for g,v in genes_to_group.items() if len(v) == 1}
    
    groups = sorted(set(genes_to_group.values()))
    groups_ix = dict(zip(groups, range(len(groups))))

    genes_to_group = { g: groups_ix[v] for g,v in genes_to_group.items() }
    return genes_to_group, groups

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
    
    
    return genes_to_complexes

if __name__ == "__main__":
    
    main('../generated-data/ppc_yeast', '../generated-data/task_yeast_gi_hybrid', 
        '../generated-data/features/ppc_yeast_smf_binned.npz', '../generated-data/yeast')
