import numpy as np 
import networkx as nx 

def main(gpath, feature_path, output_path):


    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    X = np.load(feature_path)

    F = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        a = nodes[i]

        a_neighbors = set(G.neighbors(a))
        print("%6d %a" % (i, a))
        for j in range(i+1, len(nodes)):
            b = nodes[j]

            b_neighbors = set(G.neighbors(b))

            common_neighbors = a_neighbors & b_neighbors 

            if len(common_neighbors) == 0:
                continue 

            W = np.array([ [X[i, node_ix[n]] for n in common_neighbors ], 
                           [X[j, node_ix[n]] for n in common_neighbors ]])
            max_W = np.max(W, axis=0)
            
            F[i, j] = np.max(max_W)
            F[j, i] = F[i, j]
    
    np.save(output_path, F)

if __name__ == "__main__":
    import sys 
    main(*sys.argv[1:])
