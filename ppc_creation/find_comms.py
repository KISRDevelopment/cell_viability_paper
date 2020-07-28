import igraph as ig 
import networkx as nx 
import numpy as np 
import json 
import sys
import os 
from collections import defaultdict 

def main(gpath, steps):
    
    G = ig.read(gpath)

    output = G.community_walktrap(steps=steps)
    
    if hasattr(output, 'as_clustering'):
        output = output.as_clustering()
    
    comms = { 
        G.vs['label'][i] : "comm %d" % output.membership[i]
        for i in range(G.vcount())
    }

    print("Num communities: %d" % (len(np.unique(list(comms.values())))))

    comms_to_nodes = defaultdict(set)
    for g, c in comms.items():
        comms_to_nodes[c].add(g)

    comm_sizes = np.array([len(comms_to_nodes[c]) for c in comms_to_nodes])

    print("Median communitiy size: %0.2f" % np.median(comm_sizes))
    print("# communities with 1 node: %d" % np.sum(comm_sizes == 1))
    
    output_path = '../generated-data/communities/%s_%dsteps.json' % (os.path.basename(gpath).replace('.gml',''), steps)

    with open(output_path, 'w') as f:
        json.dump(comms, f)

if __name__ == "__main__":
    gpath = sys.argv[1]
    steps = int(sys.argv[2])
    

    main(gpath, steps)
