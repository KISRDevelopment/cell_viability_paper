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

COLORS = [
    (1, 0, 0, 1),
    (1, 1, 0, 1),
    (0, 204/255, 0, 1),
    (61/255, 119/255, 1, 1)
]

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    gene_ids_to_names = json.load(f)

BINS = ['neg', 'neutral', 'positive', 'supp']

def main(task_path, go_path, output_path):

    # load GO features
    d = np.load(go_path)
    F = d['F']
    labels = np.array([gene_ids_to_names[n] for n in d['feature_labels']])

    # get the GO counts in all bins (GOxGOxBINS)
    R = np.load("../tmp/go_enrichment_matrix.npy")
    
    G = nx.Graph()
    
    selected_terms = set()
    for bin in [0, 2, 3]:

        # bin counts
        R_b = R[:, :, bin]
    
        # compute number of interactions per term
        interactions_by_term = np.sum(R_b, axis=1)

        # sort in descending order
        indecies = np.argsort(-interactions_by_term)

        # total number of interactions (double counted but it is ok)
        total = np.sum(interactions_by_term)
        thres = 0.8 * total

        # get the top interactors
        cum_sum_interactions = np.cumsum(interactions_by_term[indecies])
        ix = cum_sum_interactions < thres 

        print("Num terms in %d = %d, Prop interactions: %0.2f" % (bin, np.sum(ix), cum_sum_interactions[ix][-1] / total))

        indecies = indecies[ix]
        for ind in indecies:
            selected_terms.add(ind)
    
    R_tot = np.sum(R[:, :, [0, 2, 3]], axis=2)

    fixed_labels = { n: process_label(n) for n in labels[list(selected_terms)] }
    pos = None 
    
    for bin in [0, 2, 3]:

        # normalized bin counts
        R_b = R[:, :, bin] / R_tot

        G = nx.Graph()
        G.add_nodes_from([labels[n] for n in selected_terms])
        
        weights = []
        for a in selected_terms:
            node_a = labels[a]
            for b in selected_terms:
                if a == b:
                    continue 
                node_b = labels[b]
                weights.append(R_b[a, b])
        
        min_weight = np.percentile(weights, 50)
        max_weight = np.max(weights)

        for a in selected_terms:
            node_a = labels[a]
            for b in selected_terms:
                if a == b:
                    continue 
                node_b = labels[b]
                if R_b[a, b] > min_weight:
                    G.add_edge(node_a, node_b, weight=R_b[a, b], prop_interactions=R_b[a, b], color=COLORS[bin])

        f, ax = plt.subplots(1, 1, figsize=(30, 30))
    
        edges = list(G.edges())
        if pos is None:
            #pos = nx.spring_layout(G, k=12/np.sqrt(G.number_of_nodes()))
            pos = nx.shell_layout(G)

        nx.draw_networkx_nodes(G, pos=pos, node_color="#f7f7f7")
        nx.draw_networkx_edges(G, edgelist=edges, pos=pos, 
            width=[G[e[0]][e[1]]['prop_interactions']*5/max_weight for e in edges],
            edge_color=[G[e[0]][e[1]]['color'] for e in edges])
        nx.draw_networkx_labels(G, labels=fixed_labels, 
            pos=pos, font_family="Liberation Serif", font_size=40, font_weight='bold')
        
        # fix margins
        # https://stackoverflow.com/questions/50453043/networkx-drawing-label-partially-outside-the-box
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        y_max = max(y_values)
        y_min = min(y_values)
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        plt.savefig(output_path + BINS[bin] + '.png', bbox_inches='tight')


def process_label(s):
    s = s[0].upper() + s[1:]

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
