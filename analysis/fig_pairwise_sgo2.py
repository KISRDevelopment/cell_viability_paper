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
import scipy.stats 

BIN_LABELS = ['Negative', 'Neutral', 'Positive', 'Suppression']
COLORS = ['#FF0000', '#FFFF00', '#00CC00', '#3d77ff']

WITH_LABELS = False
DPI = 150
FONT_SIZE = 60
ALPHA = 0.05
plt.rcParams["font.family"] = "Liberation Serif"

locator = ticker.AutoLocator()

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    gene_ids_to_names = json.load(f)

def main(task_path, go_path, output_path):

    d = np.load(go_path)
    F = d['F']
    labels = [gene_ids_to_names[l] for l in d['feature_labels']]

    df = pd.read_csv(task_path)
    a_id = np.array(df['a_id'])
    b_id = np.array(df['b_id'])
    bin = np.array(df['bin'])

    n_terms = F.shape[1]
    n_terms = 4
    R = np.zeros((n_terms, n_terms, 4))

    for i in range(n_terms):
        for j in range(n_terms):
            a_term = F[a_id, i]
            b_term = F[b_id, j]
            for b in range(4):
                ix = bin == b
                R[i, j, b] = np.sum(a_term[ix] * b_term[ix])
            print("Finished %d  %d" % (i,j))
    
    # fold the array
    for b in range(4):
        
if __name__ == "__main__":
    task_path = sys.argv[1]
    feature_path = sys.argv[2]
    output_path = sys.argv[3]

    main(task_path, feature_path,  output_path)

