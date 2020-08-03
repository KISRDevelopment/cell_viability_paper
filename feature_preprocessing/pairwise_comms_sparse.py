import numpy as np 
import networkx as nx 
import sys 
import json 
import os 
from collections import defaultdict 

def main(gpath, comm_path):
    
    with open(comm_path, 'r') as f:
        nodes_to_comms = json.load(f)
    
    output_path = '../generated-data/pairwise_features/%s_comms' % (os.path.basename(comm_path).replace('.json',''))
    
    G = nx.read_gpickle(gpath)

    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))
    node_set = set(nodes)

    indecies = []
    data = []

    # handles in same comm + edge, diff comm + edge
    for src_node, tar_node in G.edges():
        f = np.zeros(3)

        same_comm = nodes_to_comms[src_node] == nodes_to_comms[tar_node]
        
        has_edge = True

        within_comm = same_comm and has_edge
        f[0] = within_comm
        cross_comm = (not same_comm) and has_edge
        f[1] = cross_comm
        f[2] = same_comm

        data.append(f)
        indecies.append((node_ix[src_node], node_ix[tar_node]))

    # group by comm
    comms_to_nodes = defaultdict(list)
    for node, comm in nodes_to_comms.items():
        comms_to_nodes[comm].append(node)

    # we handle case where in same comm + no edge
    for comm, nodes in comms_to_nodes.items():
        for i in range(len(nodes)):
            src_node = nodes[i]
            for j in range(i+1, len(nodes)):
                tar_node = nodes[j]

                if not G.has_edge(src_node, tar_node):
                    f = np.zeros(3)
                    f[2] = 1
                    data.append(f)

                    indecies.append(sorted([node_ix[src_node], node_ix[tar_node]]))
                    
    
    indecies = np.array(indecies)
    data = np.array(data)
    print(data.shape)
    
    np.savez(output_path, data=data, indecies=indecies)
    

if __name__ == "__main__":
    gpath = sys.argv[1]
    comm_path = sys.argv[2]
    
    main(gpath, comm_path)
