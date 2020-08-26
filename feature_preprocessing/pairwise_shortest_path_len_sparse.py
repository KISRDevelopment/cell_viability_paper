import numpy as np 
import networkx as nx 
import igraph as ig 
import sys
import os 
import scipy.sparse as sparse 

def main(gpath):
    
    G = ig.read(gpath)
    components = G.components()

    nxG = nx.read_gpickle(gpath.replace('.gml', ''))
    nodes = sorted(nxG.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    node_id_to_comp = np.zeros((G.vcount(), 2), dtype=int)
    Ps = []
    for i, component in enumerate(sorted(components, key=lambda c: len(c), reverse=True)):
        sG = G.subgraph(component)
        if sG.vcount() > 1:
            print("Component %d" % sG.vcount())

        P = sG.shortest_paths_dijkstra()
        for n in range(sG.vcount()):
            label = sG.vs[n]['label']
            node_n = node_ix[label]
            assert np.sum(node_id_to_comp[node_n, :]) == 0
            node_id_to_comp[node_n, :] = (i, n)

        Ps.append(P)
        
    output_path = "../generated-data/pairwise_features/%s_shortest_path_len_sparse" % (os.path.basename(gpath).replace('.gml', ''))
    
    np.savez(output_path, Ps=Ps, node_id_to_comp=node_id_to_comp)

if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath)
