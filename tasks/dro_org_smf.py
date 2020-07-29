import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import networkx as nx
from collections import defaultdict
import json 

def main(gpath):
    
    G = nx.read_gpickle(gpath)
    node_set = set(G.nodes())
    node_ix = dict(zip(sorted(G.nodes()), np.arange(len(node_set))))
    
    df_fbal_to_fbgn = pd.read_csv('../data-sources/dro/fbal_to_fbgn_fb_2020_03.tsv', sep='\t', header=1)
    fbal_to_fbgn = dict(zip(df_fbal_to_fbgn['#'], df_fbal_to_fbgn['AlleleSymbol'].str.lower()))

    df_allele = pd.read_csv('../data-sources/dro/allele_phenotypic_data_fb_2020_03.tsv', sep='\t', header=3)
    df_allele = df_allele[~pd.isnull(df_allele['allele_FBal#'])]
    df_allele['gene'] = [fbal_to_fbgn[r] for r in df_allele['allele_FBal#']]
    
    edf = pd.read_csv('../data-sources/dro/Essential genes.csv')
    edf['Gene'] = edf['Gene'].str.lower()
    essential_genes = set(edf['Gene'])
    
    df_allele = df_allele[~df_allele['gene'].isin(essential_genes) & df_allele['gene'].isin(node_ix)]
    ix = df_allele['phenotype'].str.startswith('lethal').astype(np.bool) & \
        ~df_allele['phenotype'].str.contains('with').astype(np.bool) & \
        ~pd.isnull(df_allele['phenotype']) 
    
    organism_lethal_genes = set(df_allele[ix]['gene'])
    normal_genes = node_set - essential_genes - organism_lethal_genes

    lethal_rows = [{ "gene" : g, "bin" : 0 } for g in organism_lethal_genes]
    normal_rows = [{ "gene" : g, "bin" : 1 } for g in normal_genes]
    rows = lethal_rows + normal_rows

    smf_df = pd.DataFrame(rows)
    smf_df['id'] = [node_ix[e] for e in smf_df['gene']]
    
    print([np.sum(smf_df['bin'] == b) for b in [0, 1]])

    smf_df.to_csv('../generated-data/task_dro_smf_org', index=False)
    #df_lethal = df_allele[ix]
    

    # df_lethal.to_excel('../tmp/dro_org_lethals.xlsx', index=False, columns=['gene', 'allele_FBal#', 'phenotype'])
    
    # gene_phenotypes = defaultdict(list)
    # for i, row in df_lethal.iterrows():
    #     gene_phenotypes[row['gene']].append(row['phenotype'])
    
    # print(len(gene_phenotypes))

    
    # with open('../tmp/dro', 'w') as f:
    #     json.dump(gene_phenotypes, f, indent=4)
def get_top_phenotype(s):
    parts = s.split(',')
    return parts[0].strip()

if __name__ == "__main__":
    gpath = sys.argv[1]
    main(gpath)
    