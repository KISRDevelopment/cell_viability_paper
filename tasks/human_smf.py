import pandas as pd 
import numpy as np 
import networkx as nx 
import sys 

PATH = '../data-sources/human/NIHMS732683-supplement-supp_table_3.xlsx'

def main(gpath, output_path):

    df = pd.read_excel(PATH)

    df['gene'] = [e.lower() for e in df['Gene']]

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))
    
    ix_in_net = np.isin(df['gene'], list(node_ix.keys()))

    df = df[ix_in_net]
    cs = np.array(df['KBM7 CS'])
    pval = np.array(df['KBM7 adjusted p-value'])
    
    ix_lethal = (cs < -2) & (pval < 0.05)
    ix_sick = (cs >= -2) & (cs < -1) & (pval < 0.05)
    ix_healthy = (cs >= -1)
    
    df['bin'] = -1 * ix_lethal + 1 * ix_sick + 2 * ix_healthy
    df['cs'] = cs
    df['pval'] = pval
    df['id'] = [node_ix[g] for g in df['gene']]

    df = df[df['bin'] != 0]
    df['bin'] = np.maximum(0, df['bin'])
    df.to_csv(output_path, columns=['gene', 'id', 'bin', 'pval', 'cs'], index=False)

    print("Bin counts:")
    print([np.sum(df['bin'] == b) for b in [0,1,2]])

if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath)

