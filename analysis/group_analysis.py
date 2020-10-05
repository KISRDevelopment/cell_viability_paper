import pandas as pd 
import numpy as np 
import sys
import utils.yeast_name_resolver as nr
from collections import defaultdict 
import json
import matplotlib.pyplot as plt 
import scipy.stats
import numpy.random as rng
import numpy.random as rng 

res = nr.NameResolver()

THRES = 0.25

BINS = ['I', 'N']
CONN_TYPES = ['W', 'A']
SMF_BINS = ['L', 'R', 'N']

def main(groups_file, pairs_file):

    groups = pd.read_csv(groups_file)
    group_ix = dict(zip(groups['group'], range(groups.shape[0])))
    pairs_df = pd.read_csv(pairs_file)
    
    summary = summarize_genes_examined(pairs_df, groups)

    eligible = set([g for g, s in summary.items() if 
        (len(s['genes_examined_within']) >= THRES * s['n_genes']) and 
        (len(s['genes_examined_outside']) >= THRES * groups['n_genes_glob'][0])
    ])

    a_is_eligible = pairs_df['a_group'].isin(eligible)
    b_is_eligible = pairs_df['b_group'].isin(eligible)

    print(pairs_df.shape)
    pairs_df = pairs_df[a_is_eligible & b_is_eligible]
    print(pairs_df.shape)

    s = construct_overall_summary(pairs_df)
        
    s_within = s[:,0,:,:]
    s_within_interacting_freq = s_within[0,:,:] / np.sum(s_within, axis=0, keepdims=True)
    print(s_within_interacting_freq)

    s_within = s[:,0,:,:]
    s_within_interacting_freq = s_within[0,:,:] / np.sum(s_within[0,:,:])
    print(s_within_interacting_freq)
    

    B = np.sum(s, axis=1)
    B = B[0, :, :] / np.sum(B[0, :, :])
    print(B)

def summarize_genes_examined(pairs_df, groups):

    summary = defaultdict(lambda: {
        "n_genes" : 0,
        "genes_examined" : set(),
        "genes_examined_within": set(),
        "genes_examined_across" : set(),
        "genes_examined_outside" : set()
    })

    
    for i, r in groups.iterrows():
        summary[i]['n_genes'] = r['n_genes']

    a_gene = np.array(pairs_df['a_id'])
    b_gene = np.array(pairs_df['b_id'])
    a_group = np.array(pairs_df['a_group'])
    b_group = np.array(pairs_df['b_group'])
    
    for i in range(pairs_df.shape[0]):

        if a_group[i] == b_group[i]:
            summary[a_group[i]]['genes_examined'].add(a_gene[i])
            summary[a_group[i]]['genes_examined'].add(b_gene[i])
            summary[a_group[i]]['genes_examined_within'].add(a_gene[i])
            summary[a_group[i]]['genes_examined_within'].add(b_gene[i])
        else:
            summary[a_group[i]]['genes_examined'].add(a_gene[i])
            summary[b_group[i]]['genes_examined'].add(b_gene[i])
            summary[a_group[i]]['genes_examined_across'].add(a_gene[i])
            summary[b_group[i]]['genes_examined_across'].add(b_gene[i])
            summary[a_group[i]]['genes_examined_outside'].add(b_gene[i])
            summary[b_group[i]]['genes_examined_outside'].add(a_gene[i])
            
    for g, s in summary.items():
        assert(s['n_genes'] >= len(s['genes_examined']))
    
    return summary

def construct_overall_summary(pairs_df):

    # (2 bins x 2 conn types x 3 smf x 3 smf)
    summary = np.zeros((2, 2, 3, 3))

    is_neutral = np.array(pairs_df['bin'] == 1).astype(int)
    is_across = np.array(~pairs_df['same_group']).astype(int)
    a_smf = np.array(pairs_df['a_smf']).astype(int)
    b_smf = np.array(pairs_df['b_smf']).astype(int)

    for i in range(pairs_df.shape[0]):
        s1, s2 = sorted([a_smf[i], b_smf[i]])
        summary[is_neutral[i], is_across[i], s1, s2] += 1

    return summary

if __name__ == "__main__":
    
    main('../generated-data/yeast_complexes', '../generated-data/yeast_complex_pairs')
    #main('../generated-data/yeast_pathways', '../generated-data/yeast_pathway_pairs')
    