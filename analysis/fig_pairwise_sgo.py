import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.metrics
import matplotlib.pyplot as plt 
import json
from collections import defaultdict
import matplotlib.colors
import matplotlib.ticker as ticker
import sys 

BIN_LABELS = ['Negative', 'Neutral', 'Positive', 'Suppression']
COLORS = ['#FF0000', '#FFFF00', '#00CC00', '#3d77ff']

WITH_LABELS = False
DPI = 150
FONT_SIZE = 60

plt.rcParams["font.family"] = "Liberation Serif"

locator = ticker.AutoLocator()
with open('../generated-data/go_ids_to_names.json', 'r') as f:
    gene_ids_to_names = json.load(f)

def main(task_path, go_path, output_path):

    d = np.load(go_path)
    F = d['F']
    labels = [gene_ids_to_names[l] for l in d['feature_labels']]

    counts_by_term = defaultdict(lambda: [0, 0, 0, 0])

    df = pd.read_csv(task_path)
    a_id = np.array(df['a_id'])
    b_id = np.array(df['b_id'])
    bin = np.array(df['bin'])

    n_terms = F.shape[1]

    # R = np.zeros((n_terms, n_terms, 4))

    # for i in range(df.shape[0]):
        
    #     a_terms = np.where(F[a_id[i], :])[0]
    #     b_terms = np.where(F[b_id[i], :])[0]
        
    #     pairs = set()
    #     for aid in a_terms:
    #         for bid in b_terms:
    #             pairs.add(tuple(sorted((aid, bid))))
    #     for aid, bid in pairs:
    #         R[aid, bid, bin[i]] += 1
    #         R[bid, aid, bin[i]] += 1
    
    #     if i % 10000 == 0:
    #         print(i)

    #np.save("../tmp/go_enrichment_matrix", R)
    transform = create_transform(0.5, 0.75)
    R = np.load("../tmp/go_enrichment_matrix.npy")
    
    R_tot = np.sum(R, axis=2)
    for i, bin in enumerate([0, 2, 3]):
        R_b = R[:, :, bin]
        R_b /= R_tot
        R_b[R_tot == 0] = 0
        R[:,:,bin] = R_b 
    
    CMAPS = ['Reds', '', 'YlGn', 'Blues']

    for i, bin in enumerate([0, 2, 3]):
        
        R_b = R[:,:,bin]
        
        min_R_b, max_R_b = np.min(R_b), np.max(R_b)
        R_b = (R_b - min_R_b) / (max_R_b - min_R_b)
        R_b = transform(R_b)

        cmap = CMAPS[bin]

        f, ax = plt.subplots(1, 1, figsize=(50, 50))
        r = ax.imshow(R_b, origin='lower', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(BIN_LABELS[bin], fontsize=36)

        if WITH_LABELS:
            ax.set_xticks(np.arange(R.shape[0]))
            ax.set_yticks(np.arange(R.shape[0]))
            ax.tick_params(axis='x', rotation=90)
            ax.set_xticklabels(labels, fontdict={ "fontsize" : FONT_SIZE, "weight": "bold" })
            ax.set_yticklabels(labels, fontdict={ "fontsize" : FONT_SIZE, "weight": "bold" })
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        output_path = "../tmp/go_term_enrichment_%s_%s.png" % (BIN_LABELS[bin], 'labels' if WITH_LABELS else 'nonlabels') 
        plt.savefig(output_path, bbox_inches='tight', dpi=DPI, quality=100)
    
        f, ax = plt.subplots(1, 1, figsize=(10, 20))
        xs = np.linspace(0, 1, 1000)
        ys = xs
        ax.imshow(ys[:,np.newaxis], aspect=1/60, cmap=cmap)

        print("Min %f - Max %f" % (min_R_b, max_R_b))
        xpos = locator.tick_values(min_R_b, max_R_b)
        
        xticks = transform((xpos - min_R_b) / (max_R_b - min_R_b))  * 1000
        ax.set_yticks(xticks)
        
        ax.set_yticklabels(['%0.3f' % e for e in xpos], fontsize=42, fontweight='bold')
        ax.set_ylim([0, 1000])
        ax.yaxis.tick_right()
        ax.set_xticks([])

        output_path = "../tmp/go_term_enrichment_%s_colorbar.png" % (BIN_LABELS[bin]) 
        plt.savefig(output_path, bbox_inches='tight', dpi=150, quality=100)
        
        rows = []
        for a in range(R.shape[0]):
            for b in range(a, R.shape[0]):
                rows.append((labels[a], labels[b], R_b[a, b]))

        sorted_rows = sorted(rows, key=lambda t: t[2], reverse=True)

        print(bin)

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
    feature_path = sys.argv[2]
    output_path = sys.argv[3]

    main(task_path, feature_path,  output_path)

