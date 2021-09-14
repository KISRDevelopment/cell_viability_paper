import numpy as np 
import pyreadr 
import networkx as nx
import utils.yeast_name_resolver
import scipy.stats  as stats
import os 

res = utils.yeast_name_resolver.NameResolver()

def main(gpath, r_features_path):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    result = pyreadr.read_r(r_features_path)
    F = result['F']
    n_features = F.shape[1]
    
    genes = res.get_unified_names(F.index)
    
    assert len(set(genes)) == F.shape[0]

    F['gene'] = genes
    F = F.set_index('gene')
    
    output_F = np.zeros((len(nodes), n_features))
    
    common_nodes = set(nodes) & set(genes)

    for i, node in enumerate(nodes):
        if node in common_nodes:
            output_F[i, :] = F.loc[node]
    
    mu = np.mean(output_F, axis=0)
    std = np.std(output_F, axis=0)

    print(mu)
    print(std)

    # normalize
    output_F = stats.zscore(output_F, axis=0)
    

    print(output_F.shape)
    
    feature_labels = list(F.columns)
    assert len(feature_labels) == output_F.shape[1]

    output_path = '../generated-data/features/%s_amino_acid' % (os.path.basename(gpath))
    np.savez(output_path, F=output_F, feature_labels=feature_labels, mu=mu, std=std)


if __name__ == "__main__":
    import sys 
    main(*sys.argv[1:])
