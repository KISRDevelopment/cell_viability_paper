import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import json
import os 
import sys

def main(gpath, sgo_files):
    
    main_sgo_file, other_sgo_files = sgo_files[0], sgo_files[1:]
    
    G = nx.read_gpickle(gpath)
    nodes = list(sorted(G.nodes()))
    nodes_set = set(nodes)
    node_ix = dict(zip(nodes, range(len(nodes))))

    d = np.load(main_sgo_file)
    F = d['F']
    
    labels = d['feature_labels'].tolist()
    label_ids = dict(zip(labels, range(len(labels))))

    label_sets = [set(labels)]
    for file in other_sgo_files:
        d = np.load(file)
        label_sets.append(set(d['feature_labels'].tolist()))
    
    common_labels = sorted(set.intersection(*label_sets))

    print("Common: %d" % len(common_labels))
    
    lids = [label_ids[l] for l in common_labels]

    F = F[:, lids]
    print(F.shape)
    output_path = '../generated-data/features/%s_common_sgo' % (os.path.basename(gpath))
        
    np.savez(output_path, F=F, feature_labels=common_labels)

if __name__ == "__main__":
    gpath = sys.argv[1]
    main_sgo_file = sys.argv[2]
    sgo_files = sys.argv[3:]

    main(gpath, [main_sgo_file]+ sgo_files)
    