import numpy as np 
import pandas as pd 
import networkx as nx 
import feature_preprocessing.redundancy as red 
import sys 
import os
def main(gpath, blastp):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    results = red.read_blastp_results('../tmp/blastp_yeast', red.yeast_get_name)

    F = np.zeros((len(nodes), len(nodes)))
    for q in results:
        targets = results[q]
        subjects = [t['subject'] for t in targets]

        for s in subjects:
            if q in node_ix and s in node_ix:
                q_ix = node_ix[q]
                s_ix = node_ix[s]
                F[q_ix, s_ix] = 1
                F[s_ix, q_ix] = 1
    
    print(np.sum(F))
    output_path = "../generated-data/pairwise_features/%s_homology" % (os.path.basename(gpath))
    
    np.save(output_path, F)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
