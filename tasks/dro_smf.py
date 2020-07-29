import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import networkx as nx

edf = pd.read_csv('../data-sources/dro/Essential genes.csv')
edf['Gene'] = edf['Gene'].str.lower()
essential_genes = set(edf['Gene'])

def main(gpath):
    
    G = nx.read_gpickle(gpath)
    node_set = set(G.nodes())
    node_ix = dict(zip(sorted(G.nodes()), np.arange(len(node_set))))
    
    print("Essentials in graph: %d" % len(essential_genes.intersection(node_set)))

    df = pd.read_excel('../data-sources/dro/elife-36333-supp1-v1.xlsx', sheet_name='Computed MLE result')
    df['Gene'] = df['Gene'].str.lower()

    r1 = df['Rep1']
    r2 = df['Rep2']
    mu = df['sgRNA-level Average']

    lethal_ix = df['Gene'].isin(essential_genes)
    std = np.std(np.vstack((r1, r2)).T, axis=1, ddof=1)
    p = 1-stats.norm.cdf(0.0, mu, std) 
    healthy_ix = ~lethal_ix & (p >= 0.2)
    sick_ix = ~lethal_ix & ~healthy_ix

    print(np.sum(lethal_ix))
    print(np.sum(sick_ix))
    print(np.sum(healthy_ix))

    gene = df['Gene']

    lethal = [{ "gene" : e[0], "is_lethal" : 1, "bin" : 0, "cs" : e[1] } for e in 
        zip(gene[lethal_ix], mu[lethal_ix])]
    sick = [{ "gene" : e[0], "is_lethal" : 0, "bin" : 1, "cs" : e[1] } for e in 
        zip(gene[sick_ix], mu[sick_ix])]
    healthy = [{ "gene" : e[0], "is_lethal" : 0, "bin" : 2, "cs" : e[1] } for e in 
        zip(gene[healthy_ix], mu[healthy_ix])]
    rows = lethal + sick + healthy

    smf_df = pd.DataFrame(rows)
    smf_df = smf_df[smf_df['gene'].isin(node_set)]

    smf_df['id'] = [node_ix[e] for e in smf_df['gene']]
    
    print(smf_df.shape)

    print(np.sum(smf_df['bin'] == 0))
    print(np.sum(smf_df['bin'] == 1))
    print(np.sum(smf_df['bin'] == 2))

    smf_df.to_csv('../generated-data/task_dro_smf', index=False)
    
if __name__ == "__main__":
    gpath = sys.argv[1]
    main(gpath)
    