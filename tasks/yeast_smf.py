import os
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import networkx as nx 
import sys 

from utils import yeast_name_resolver

res = yeast_name_resolver.NameResolver()


strain_priorties = [ ('dma',),
                     ('sn',)]

DATASET_PATH = '../data-sources/yeast/strain_ids_and_single_mutant_fitness.xlsx'
ESSENTIALS_PATH = '../data-sources/yeast/costanzo2016/SGA_ExE.txt'

# min probability of >= 1 to be considered normal
ALPHA = 0.2
CUTOFF = 1.0

def main(gpath, temp, output_path):
    
    G = nx.read_gpickle(gpath) 
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))
    
    df = pd.read_excel(DATASET_PATH)
    
    create_dataset(df, node_ix, True, temp, output_path)
    
def create_dataset(df, node_ix, include_lethals, temp, path):
    
    nodes = set(node_ix.keys())
    print()
    print("Creating dataset for temp: %d" % temp)
    all_smf = {}
    rem_genes = nodes
    for strains in strain_priorties:
        smf = extract_data(df, rem_genes, strains, temp, include_lethals)
        collected_genes = set(smf.keys())
        rem_genes = rem_genes.difference(collected_genes)
        print("Strains %s: %d" % (' '.join(strains), len(collected_genes)))
        all_smf = { **all_smf, **smf }
    
    df = create_df(all_smf)
    df = add_lethals(df)
    
    lethals_ix = df['is_lethal'] == 1
    print("Lethals: %d, Nonlethals: %d, Total: %d" % (np.sum(lethals_ix), np.sum(~lethals_ix), df.shape[0]))
    
    # need to do this because addlethals adds things not in node_ix
    df = df[df['gene'].isin(node_ix)]
    df['id'] = [node_ix[g] for g in df['gene']]

    # bin the observations
    cs = np.array(df['cs'])
    std = np.array(df['std'])
    nonlethal_ix = cs > 0
    prob_healthy = (1 - stats.norm.cdf(CUTOFF, cs[nonlethal_ix], std[nonlethal_ix])) >= ALPHA
    bins = np.zeros(df.shape[0])
    bins[nonlethal_ix] = prob_healthy + 1
    df['bin'] = bins

    print("Lethals: %d, Nonlethals: %d" % (np.sum(df['is_lethal']), np.sum(1-df['is_lethal'])))
    print([np.sum(bins == b) for b in np.unique(bins)])
    
    df.to_csv(path, index=False)

def extract_data(df, nodes, target_strains, temp, include_lethals):
    
    smf = {}
    
    for i, row in df.iterrows():
        
        gene = row['Systematic gene name']
        gene = res.get_unified_name(gene)
        
        strain = row['Strain ID'].split('_')[1]
        strain = re.sub(r'\d+', '', strain)
        
        if target_strains and strain not in target_strains:
            continue
        
        if gene in nodes:
            
            mu = row['Single mutant fitness (%d°)' % temp]
            std = row['Single mutant fitness (%d°) stddev' % temp]
            
            if include_lethals and np.isnan(mu):
                mu = 0.
                std = 0.
            elif not include_lethals and np.isnan(mu):
                continue
            
            if gene not in smf:
                smf[gene] = { 'means' : [], 'stds' : [] }
                
            smf[gene]['means'].append(mu)
            smf[gene]['stds'].append(std)
    
    return smf
    
def create_df(smf):
    new_smf = [
        { 
            'gene' : k,
            **add_normals(**v)
        }
        for k,v in smf.items()
    ]
    df = pd.DataFrame(new_smf)
    return df 

def add_lethals(df):

    essentials = get_essential_genes(ESSENTIALS_PATH)

    smf_genes = list(df['gene'])
    smf_cs = list(df['cs'])
    smf_std = list(df['std'])

    nodes_to_smf = dict(zip(smf_genes, smf_cs))
    nodes_to_std = dict(zip(smf_genes, smf_std))

    # there are genes reported with 0 cs but are not in the essentials list, we drop those
    lethals = essentials
    unknown_genes = set([g for g in smf_genes if nodes_to_smf[g] == 0 and g not in essentials])
    print("Unknown: %d" % len(unknown_genes))
    nonlethals = set(smf_genes) - lethals - unknown_genes

    lethal_rows = [{ 'gene' : g, 'is_lethal' : 1, 'cs' : 0, 'std' : 0 } for g in lethals]
    nonlethal_rows = [{ 'gene' : g, 'is_lethal' : 0, 'cs' : nodes_to_smf[g], 'std' : nodes_to_std[g] } for g in nonlethals]
    
    rows = nonlethal_rows + lethal_rows
    df = pd.DataFrame(rows)
    
    return df 

def get_essential_genes(path):
    df = pd.read_csv(path, sep='\t')
    queries = [e.split('_')[0] for e in df['Query Strain ID']]
    arrays = [e.split('_')[0] for e in df['Array Strain ID']]
    all = list(set(queries + arrays))
    all = res.get_unified_names(all)
    return set(all)

def add_normals(means, stds):
    N = len(means)
    variances = np.power(stds,2)

    new_mean = np.sum(means) / N
    new_var = np.sum(variances) / np.power(N,2)
    new_std = np.sqrt(new_var)
    
    if new_std > 0:
        a, b = stats.norm.interval(0.95, loc=new_mean, scale=new_std)
    else:
        a, b = 0., 0.
        
    return { 
        'cs' : new_mean, 
        'std' : new_std,
        'N' : N,
        'lower' : a,
        'upper' : b
    }


    
if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath)
    