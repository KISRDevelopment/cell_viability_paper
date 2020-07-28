import numpy as np 
import networkx as nx 
import sys
import os 

def main(gpath):
    
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    nodes_ix = dict(zip(nodes, np.arange(len(nodes))))

    D = nx.shortest_path_length(G)

    F = np.zeros((len(nodes), len(nodes)))
    for src, targets in D:
        src_ix = nodes_ix[src]
        for target, plen in targets.items():
            target_ix = nodes_ix[target]
            F[src_ix, target_ix] = plen 
            F[target_ix, src_ix] = plen 
    
    output_path = "../generated-data/pairwise_features/%s_shortest_path_len" % (os.path.basename(gpath))
    
    np.save(output_path, F)

if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath)
