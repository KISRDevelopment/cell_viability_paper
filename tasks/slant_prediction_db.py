import numpy as np 
import pandas as pd 
import networkx as nx 
import sys 

def main(org, gpath, slant_path, output_path):

    df = pd.read_csv(slant_path, sep='\t', header=0, names=['a', 'b', 'official_a', 'official_b', 'organism', 'source', 'ref', 'quality'])
    df['a'] = df['a'].str.lower()
    df['b'] = df['b'].str.lower()
    df['official_a'] = df['official_a'].str.lower()
    df['official_b'] = df['official_b'].str.lower()

    G = nx.read_gpickle(gpath)
    if org == 'yeast':
        nodes = [n.split('  ')[0] for n in sorted(G.nodes())]
    else:
        nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))
    print("Nodes: %d" % len(node_ix))

    a_col = 'official_a'
    b_col = 'official_b'
    if org == 'human':
        a_col, b_col = 'a', 'b'
    
    ix_matching = df[a_col].isin(node_ix) & df[b_col].isin(node_ix)
    print("Matching: %d out of %d" % (np.sum(ix_matching), df.shape[0]))

    df =  df[ix_matching]
    df['a_id'] = [node_ix[n] for n in df[a_col]]
    df['b_id'] = [node_ix[n] for n in df[b_col]]

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    org = sys.argv[1]
    gpath = sys.argv[2]
    slant_path = sys.argv[3]
    output_path = sys.argv[4]

    main(org, gpath, slant_path, output_path)