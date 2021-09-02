#
# implementation of ACDD metric from Alanis-Lobato et al 2013
#
import pandas as pd 
import numpy as np 
import networkx as nx 
import os 

def main(gpath):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    # calculate average degree
    degrees = list(G.degree())
    navg = np.mean([v[1] for v in degrees])
    print("Average degree: %0.2f" % navg)

    F = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        node_i = nodes[i]

        Gamma_x = set(G.neighbors(node_i))
        gamma_x = Gamma_x | set([node_i])
        lambda_x = max(0, navg - len(gamma_x))

        for j in range(i+1, len(nodes)):
            node_j = nodes[j]
            
            Gamma_y = set(G.neighbors(node_j))
            gamma_y = Gamma_y | set([node_j])
            lambda_y = max(0, navg - len(gamma_y))

            union = gamma_x | gamma_y 
            intersect = gamma_x & gamma_y

            sym_diff = union - intersect

            acdd = (len(sym_diff) + lambda_x + lambda_y) / (len(union) + len(intersect))
            F[i, j] = acdd 
            F[j, i] = acdd 

        print("Finished %d" % i)

    output_path = "../generated-data/pairwise_features/%s_acdd" % (os.path.basename(gpath))
    
    np.save(output_path, F) 





if __name__ == "__main__":
    import sys 
    main(sys.argv[1])

