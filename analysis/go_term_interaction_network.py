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
    labels = np.array([gene_ids_to_names[n] for n in d['feature_labels']])

    transform = create_transform(0.5, 0.75)
    R = np.load("../tmp/go_enrichment_matrix.npy")
    
    R_tot = np.sum(R[:, :, [0, 1, 2, 3]], axis=2)

    G = nx.Graph()
    
    bin = selected_bin


    R_b = R[:, :, bin].copy()
    #R_b /= R_tot
    #R_b[R_tot == 0] = 0

    #R[:,:,bin] = R_b 

    n_total_bin = np.sum(np.triu(R_b, 1)) + np.sum(np.diagonal(R_b))
    print("Num interactions in bin: %d" % n_total_bin)

    interactions_by_term = np.sum(R_b, axis=1)
    
    ix = np.argsort(-interactions_by_term)
    
    total = np.sum(interactions_by_term)
    thres = 0.8
    sorted_interactions_by_term = interactions_by_term[ix] 
    n_interactions = 0
    for i in range(0, len(ix)):
        n_interactions += sorted_interactions_by_term[i]
        print("top %d, n_interactions=%d, prop=%0.2f" % (i, n_interactions, n_interactions / total))
        prop = n_interactions / total
        if prop >= thres:
            break

    ix = ix[:i]
    R_b = R_b[ix, :]
    R_b = R_b[:, ix]
    labels = labels[ix]

    rows = []
    for a in range(R_b.shape[0]):
        for b in range(a, R_b.shape[0]):
            rows.append((labels[a], labels[b], R_b[a, b], R_b[a, b]))

    interaction_distrib = [r[3] for r in rows]

    median = np.percentile(interaction_distrib, 0)
    median_weight = np.percentile([r[2]  for r in rows], 50)
    max_weight = np.max([r[2] for r in rows])
    
    covered_interactions = 0
    for a, b, weight, tot in rows:
        if a == b:
            continue 
        
        if tot < median:
            continue 
        
        if weight < median_weight:
            continue 
        
        covered_interactions += weight

        #weight = transform(weight)
        G.add_edge(a, b, 
            weight=weight*10/max_weight, penwidth=weight*2, 
            rank=2,
            bin=bin, color=COLORS[bin])

    print("Covered: %d (%0.2f)" % (covered_interactions, covered_interactions / n_total_bin))

    for n in G.nodes():
        G.nodes[n]['fontsize'] = 28
        G.nodes[n]['color'] = "black"
        G.nodes[n]['fillcolor'] = 'white'
        G.nodes[n]['style'] = '"filled"'
        G.nodes[n]['shape'] = 'box'
        G.nodes[n]['rank'] = 1
    
    fixed_labels = { n: break_label(n) for n in G.nodes() }
    #print(nx.info(G))
    #write_dot(G, output_path)
    
    f, ax = plt.subplots(1, 1, figsize=(25, 20))
    
    edges = list(G.edges())
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color="#f7f7f7")
    nx.draw_networkx_edges(G, edgelist=edges, pos=pos, 
        width=[G[e[0]][e[1]]['weight'] for e in edges],
        edge_color=[G[e[0]][e[1]]['color'] for e in edges])
    nx.draw_networkx_labels(G, labels=fixed_labels, pos=pos, font_size=20, font_weight='bold')
    plt.savefig(output_path, bbox_inches='tight')


    #plt.show()

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

def break_label(s):

    if len(s) < 8:
        return s 
    
    parts = s.split(' ')
    
    n = len(parts)
    if n == 1:
        return s
    
    half_n = n // 2

    return "%s\n%s" % (" ".join(parts[:half_n]), " ".join(parts[half_n:]))




if __name__ == "__main__":
    task_path = sys.argv[1]
    go_path = sys.argv[2]
    output_path = sys.argv[3] 
    main(task_path, go_path, output_path)
