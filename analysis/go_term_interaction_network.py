import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.metrics
import matplotlib.pyplot as plt 
import json
from collections import defaultdict
import matplotlib.colors
import networkx as nx 
from networkx.drawing.nx_pydot import write_dot
import sys 

BIN_LABELS = ['Negative', 'Neutral', 'Positive', 'Suppression']
COLORS = ['#FF0000', '#FFFF00', '#00CC00', '#3d77ff']
with open('../generated-data/go_ids_to_names.json', 'r') as f:
    gene_ids_to_names = json.load(f)
    
def main(task_path, go_path, output_path, selected_bin=3):


    d = np.load(go_path)
    F = d['F']
    labels = [gene_ids_to_names[n] for n in d['feature_labels']]

    transform = create_transform(0.5, 0.75)
    R = np.load("../tmp/go_enrichment_matrix.npy")
    
    R_tot = np.sum(R[:, :, [0, 1, 2, 3]], axis=2)

    G = nx.Graph()
    
    bin = selected_bin

    R_b = R[:, :, bin]
    R_b /= R_tot
    R_b[R_tot == 0] = 0
    R[:,:,bin] = R_b 

    print(np.max(R_tot))
    print(np.min(R_tot))
    
    rows = []
    for a in range(R.shape[0]):
        for b in range(a, R.shape[0]):
            #print("[%3d] and [%3d]: %8d, prop in bin: %4.2f" % (a, b, R_tot[a, b], R_b[a,b]))
            rows.append((labels[a], labels[b], R_b[a, b], R_tot[a, b]))

    interaction_distrib = [r[3] for r in rows]
    median = np.percentile(interaction_distrib, 50)
    median_weight = np.percentile([r[2]  for r in rows], 75)

    for a, b, weight, tot in rows:
        if a == b:
            continue 
        
        if tot < median:
            continue 
        
        if weight < median_weight:
            continue 
    
        #weight = transform(weight)
        G.add_edge(a, b, 
            weight=weight*100, penwidth=weight*2, 
            rank=2,
            bin=bin, color=COLORS[bin])
            
    for n in G.nodes():
        G.nodes[n]['fontsize'] = 28
        G.nodes[n]['color'] = "black"
        G.nodes[n]['fillcolor'] = 'white'
        G.nodes[n]['style'] = '"filled"'
        G.nodes[n]['shape'] = 'box'
        G.nodes[n]['label'] = n
        G.nodes[n]['rank'] = 1
    
    #print(nx.info(G))
    #write_dot(G, output_path)
    
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    edges = list(G.edges())
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos=pos)
    nx.draw_networkx_edges(G, edgelist=edges, pos=pos, 
        width=[G[e[0]][e[1]]['weight'] for e in edges],
        edge_color=[G[e[0]][e[1]]['color'] for e in edges])
    nx.draw_networkx_labels(G, pos=pos)
    plt.show()

def create_transform(m, c):

    beta = -1
    for i in range(100):
        beta = np.log(1 - c + c * np.exp(beta)) / m 
    alpha = 1 / (1-np.exp(beta))

    print("Alpha = %0.4f, Beta=%0.4f" % (alpha,beta))
    def transform(A):

        A = alpha * (1 - np.exp(beta * A))
        return A 

    return transform

if __name__ == "__main__":
    task_path = sys.argv[1]
    go_path = sys.argv[2]
    output_path = sys.argv[3] 
    main(task_path, go_path, output_path)
