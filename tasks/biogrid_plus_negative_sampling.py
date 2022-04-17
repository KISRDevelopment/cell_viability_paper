import pandas as pd
import numpy as np
import networkx as nx
import numpy.random as rng 

def main(gpath, biogrid_path, smf_task_path, output_path, n_samples=1000000, with_smf_only=True):
    
    biogrid_df = pd.read_csv(biogrid_path)
    biogrid_pairs = to_pairs(biogrid_df)

    G = nx.read_gpickle(gpath)
    nodes = np.array(sorted(G.nodes()))
    node_ix = dict(zip(nodes, np.arange(len(nodes))))
    
    if with_smf_only:
        smf_df = pd.read_csv(smf_task_path)
        nodes = list(set(smf_df['gene']))
        print("# nodes with SMF: %d" % len(nodes))
        genes_with_smf = list(set(smf_df['id']))

    N = len(nodes)
    combs = N*(N-1)/2
    print("# possible combs: %d" % combs)
    # add neutrals
    # anything that biogrid does not classify as interaction

    a_genes = rng.permutation(n_samples + 10000) % len(nodes)
    b_genes = rng.permutation(n_samples + 10000) % len(nodes)
    ix = a_genes != b_genes
    a_genes = a_genes[ix]
    b_genes = b_genes[ix]

    rand_pairs = set([tuple(sorted((nodes[a], nodes[b]))) for a, b in zip(a_genes, b_genes)])
    rand_pairs -= biogrid_pairs
    rows = [{ "a" : a, "b" : b, "bin" : 1 } for a, b in rand_pairs][:n_samples]

    df = biogrid_df.append(pd.DataFrame(rows))

    ix = df['a'].isin(node_ix) & df['b'].isin(node_ix)
    df = df[ix]
    df['a_id'] = [node_ix[e] for e in df['a']]
    df['b_id'] = [node_ix[e] for e in df['b']]

    ix = df['a_id'] != df['b_id']
    df = df[ix]

    print("Data size: ", df.shape)
    print("Bin counts:")
    print([np.sum(df['bin'] == b) for b in [0,1,2,3]])

    if with_smf_only:
        ix_no_smf_either = ~df['a_id'].isin(genes_with_smf) | ~df['b_id'].isin(genes_with_smf)
        df = df[~ix_no_smf_either]
        print("After filtering out pairs without SMF:")
        print("Data size: ", df.shape)
        print("Bin counts:")
        print([np.sum(df['bin'] == b) for b in [0,1,2,3]])

    df.to_csv(output_path, index=False)
    
def to_pairs(df):
    a = list(df['a'])
    b = list(df['b'])
    pairs = set()
    for i in range(df.shape[0]):
        pairs.add(tuple(sorted((a[i], b[i]))))
    return pairs

if __name__ == "__main__":
    main()
    