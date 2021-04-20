import pandas as pd 
import numpy as np 
import networkx as nx 
import sys 
from utils import yeast_name_resolver

res = yeast_name_resolver.NameResolver()

TGI_PATH = "../data-sources/yeast/aao1729_Data_S1.tsv"

def main(gpath, output_path):
    
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    df = pd.read_csv(TGI_PATH, sep='\t')
    df = df[df['Combined mutant type'] == 'trigenic']

    neg_interacting_ix = (df['P-value'] < 0.05) & (np.abs(df['Adjusted genetic interaction score (epsilon or tau)']) > 0.08)
    neutral_ix = ~neg_interacting_ix

    neg_interacting_df = df[neg_interacting_ix]
    neg_triplets = get_triplets(neg_interacting_df)
    
    neutral_df = df[neutral_ix]
    neutral_triplets = get_triplets(neutral_df)

    print("Negative: %d, Neutral: %d" % (len(neg_triplets), len(neutral_triplets)))
    
    neg_rows = create_rows(neg_triplets, node_ix, 0)
    netural_rows = create_rows(neutral_triplets, node_ix, 1)

    
    print("Negative: %d, Neutral: %d" % (len(neg_rows), len(netural_rows)))
    
    df = pd.DataFrame(neg_rows + netural_rows)

    df.to_csv(output_path, index=False)
    
def create_rows(triplets, node_ix, bin):

    rows = []
    for t in triplets:

        if t[0] in node_ix and t[1] in node_ix and t[2] in node_ix:
            rows.append({
                "a" : t[0],
                "b" : t[1],
                "c" : t[2],
                "a_id" : node_ix[t[0]],
                "b_id" : node_ix[t[1]],
                "c_id" : node_ix[t[2]],
                "bin" : bin
            })
        else:
            #print("Combination is not in PPC: ")
            #print(t)
            pass
    return rows 

def get_triplets(df):
    
    qsid = list(df['Query strain ID'])
    asid = list(df['Array strain ID'])

    triplets = []
    for q, a in zip(qsid, asid):

        # remove strain identifier
        genes, strain = q.split('_')
        gene_1, gene_2 = genes.split('+')
        gene_3, strain = a.split('_')

        genes = res.get_unified_names((gene_1, gene_2, gene_3))
        
        unresolved = res.get_unresolved_names()
        if len(unresolved) > 0:
            print("Unresolved:")
            print(unresolved)
        triplets.append(genes)
    
    return triplets

if __name__ == "__main__":
    gpath = sys.argv[1]
    output_path = sys.argv[2]

    main(gpath, output_path)
