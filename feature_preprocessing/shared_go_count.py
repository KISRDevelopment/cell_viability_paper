import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import json
import os 
import sys
import obonet 

from utils import yeast_name_resolver

res = yeast_name_resolver.NameResolver()

GO_FILE = "../data-sources/yeast/sgd.gaf"


GO_GRAPH = obonet.read_obo('../tools/go.obo')
id_to_aspect = {id_: data.get('namespace') for id_, data in GO_GRAPH.nodes(data=True)}
aspects = set(id_to_aspect.values())
aspect_idx = dict(zip(aspects, range(1, len(aspects) + 1)))

def main(gpath, go_feature_file):
    
    G = nx.read_gpickle(gpath)
    nodes = list(sorted(G.nodes()))
    nodes_set = set(nodes)

    d = np.load(go_feature_file)
    terms = d['feature_labels']
    goF = d['F']

    # map the terms to functions
    aspect_vec = np.array([aspect_idx[ id_to_aspect[t] ] for t in terms])
    goF = goF * aspect_vec


    F = np.zeros((len(nodes), len(nodes), 3))
    for i in range(len(nodes)):
        
        for j in range(i+1, len(nodes)):
            
            for k in range(len(aspects)):
                F[i, j, k] = np.sum((goF[i, :] == (k+1)) & (goF[j, :] == (k+1)))
                
            F[j, i, :] = F[i, j, :]
        
            #print(F[i, j,:])
        print("Completed %d" % i)
        #break
    output_path = "../generated-data/pairwise_features/%s_shared_go" % (os.path.basename(gpath))
    
    np.save(output_path, F)

if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath, sys.argv[2])
    