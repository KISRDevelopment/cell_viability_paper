import feature_preprocessing.complexes_pathways 
import numpy as np 
import networkx as nx 
import os 

gpath = "../generated-data/ppc_yeast"

def main():

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    genes_to_pathways = feature_preprocessing.complexes_pathways.parse_kegg_pathways()

    F = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        a = nodes[i]

        if a not in genes_to_pathways:
            continue 
        
        pathways_a = genes_to_pathways[a]
        for j in range(i+1, len(nodes)):
            b = nodes[j]

            if b in genes_to_pathways:
                pathways_b = genes_to_pathways[b]

                F[i, j] = len(pathways_a & pathways_b) > 0
                F[j, i] = F[i, j]
    
    print(np.sum(F))
    print(F)
    output_path = "../generated-data/pairwise_features/%s_pathway_comembership" % (os.path.basename(gpath))
    
    np.save(output_path, F)

if __name__ == "__main__":
    main()
