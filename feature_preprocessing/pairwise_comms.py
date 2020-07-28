import numpy as np 
import networkx as nx 
import sys 
import json 
import os 

def main(gpath, comm_path):
    
    with open(comm_path, 'r') as f:
        nodes_to_comms = json.load(f)
    
    output_path = '../generated-data/pairwise_features/%s_%s_comms' % (os.path.basename(gpath), os.path.basename(comm_path).replace('.json',''))
    
    G = nx.read_gpickle(gpath)

    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))
    node_set = set(nodes)

    N = len(nodes)
    F = np.zeros((N, N, 3))
    i = 0
    for i in range(N):
        src_node = nodes[i]
        for j in range(i+1, N):
            tar_node = nodes[j]

            same_comm = nodes_to_comms[src_node] == nodes_to_comms[tar_node]
            has_edge = G.has_edge(src_node, tar_node)

            within_comm = same_comm and has_edge
            F[i, j, 0] = within_comm

            cross_comm = (not same_comm) and has_edge
            F[i, j, 1] = cross_comm

            F[i,j, 2] = same_comm

            F[j,i,0] = F[i,j,0]
            F[j,i,1] = F[i,j,1]
            F[j,i,2] = F[i,j,2]

    print("# pairs within comm: %d" % np.sum(F[:,:,0]))
    print("# pairs cross comm: %d" % np.sum(F[:,:,1]))
    print("# pairs in same comm: %d" % (np.sum(F[:,:,2])))
    
    np.save(output_path, F)

if __name__ == "__main__":
    gpath = sys.argv[1]
    comm_path = sys.argv[2]
    
    main(gpath, comm_path)
