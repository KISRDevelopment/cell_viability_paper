import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.metrics
import json
import sys 
from collections import defaultdict

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    gene_ids_to_names = json.load(f)

def main(task_file, go_file, output_path):

    d = np.load(go_file)
    F = d['F'].astype(bool)
    terms = d['feature_labels']

    df = pd.read_csv(task_file)
    bins = np.array(df['bin'])
    pair_ids = zip(df['a_id'], df['b_id'], bins)

    counts_by_term = defaultdict(lambda: [0, 0, 0, 0])

    for a_id, b_id, thebin in pair_ids:
        
        gene_terms_a = [gene_ids_to_names[gt] if gt in gene_ids_to_names else gt for gt in terms[F[a_id,:]]]
        gene_terms_b = [gene_ids_to_names[gt] if gt in gene_ids_to_names else gt for gt in terms[F[b_id,:]]]
        gene_terms = set(gene_terms_a).union(gene_terms_b)

        for gt in gene_terms:
            counts_by_term[gt][int(thebin)] += 1
    
    rows = []
    for term in counts_by_term:
        cnts = np.array(counts_by_term[term])
        normed_cnts = cnts / np.sum(cnts)

        rows.append({
            "term" : term,
            "negative" : cnts[0],
            "neutral" : cnts[1],
            "positive" : cnts[2],
            "suppression" : cnts[3],

            "negative_p" : normed_cnts[0],
            "neutral_p" : normed_cnts[1],
            "positive_p" : normed_cnts[2],
            "suppression_p" : normed_cnts[3],

            "gi" : cnts[0] + cnts[2] + cnts[3],
            "gi_p" : normed_cnts[0] + normed_cnts[2] + normed_cnts[3],

            "total" : np.sum(cnts)
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)


if __name__ == "__main__":
    task_file = sys.argv[1]
    go_file = sys.argv[2]
    output_path = sys.argv[3]

    main(task_file, go_file, output_path)
