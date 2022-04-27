import numpy as np 
import pandas as pd 
import networkx as nx 
import scipy.stats as stats
import os 

# based on supp table S2 in mistry et al. 2017
BETA = 0.3 
OMEGA = 0.9

def main(gpath, rma_corr_path):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

     
    # put coexpr correlation on edges
    # and ECC
    # rma_corr = np.load(rma_corr_path)
    # for i in range(rma_corr.shape[0]):
    #     for j in range(i+1, rma_corr.shape[1]):
    #         if G.has_edge(nodes[i], nodes[j]):
    #             G[nodes[i]][nodes[j]]['coexpr'] = rma_corr[i, j]

    #             numerator = len(list(nx.common_neighbors(G, nodes[i], nodes[j]))) + 1
    #             denominator = min(G.degree(nodes[i]), G.degree(nodes[j]))
    #             ecc = (numerator / (denominator * 1.0))
    #             G[nodes[i]][nodes[j]]['ecc'] = ecc 
    
    # nx.write_gpickle(G, "../tmp/ppc_test")
    G = nx.read_gpickle('../tmp/ppc_test')

    # calculate diffslc
    evcent = nx.eigenvector_centrality_numpy(G)

    F = np.zeros((len(nodes), 1))
    for i, node in enumerate(nodes):
        
        eigcent = evcent[node]

        bdc = np.sum([BETA * G[node][neighbor]['coexpr'] + (1-BETA) * G[node][neighbor]['ecc'] for neighbor in G.neighbors(node) if neighbor != node])
        diffslc = OMEGA * eigcent + (1-OMEGA) * bdc 

        F[i, 0] = diffslc
    
    mu = np.mean(F, axis=0)
    std = np.std(F, axis=0)

    print(mu)
    print(std)

    # normalize
    F = stats.zscore(F, axis=0)
    
    print(np.min(F, axis=0))
    print(np.max(F, axis=0))
    print(np.mean(F, axis=0))
    print(np.std(F, axis=0))

    print("Nan: %d" % np.sum(np.isnan(F)))

    output_path = '../generated-data/features/%s_diffslc' % (os.path.basename(gpath))
    np.savez(output_path, F=F, feature_labels=['diffslc'], mu=mu, std=std)


    
if __name__ == "__main__":
    import sys 
    main(sys.argv[1], sys.argv[2])
