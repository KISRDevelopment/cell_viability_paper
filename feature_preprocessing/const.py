import numpy as np 
import networkx as nx 
import sys 
import json 
import os 

def main(gpath):
    
    G = nx.read_gpickle(gpath)

    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))
    node_set = set(nodes)

    F = np.zeros((len(nodes), 1))
    np.savez("../generated-data/features/%s_const" % (os.path.basename(gpath)), F=F, feature_labels=['const']) 

    F = np.zeros((len(nodes), len(nodes), 1))
    np.save("../generated-data/pairwise_features/%s_const" % (os.path.basename(gpath)), F) 

if __name__ == "__main__":
    gpath = sys.argv[1]
    
    main(gpath)
