import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.metrics
import json
import sys 
import scipy.stats 
with open('../generated-data/go_ids_to_names.json', 'r') as f:
    gene_ids_to_names = json.load(f)

def main(task_file, go_file, output_path):

    d = np.load(go_file)
    F = d['F'].astype(bool)
    terms = d['feature_labels']

    df = pd.read_csv(task_file)
    bins = np.array(df['bin'])
    ids = df['id']

    counts_by_term = {}

    for i, gene_id in enumerate(ids):
        thebin = bins[i]

        gene_terms = [gene_ids_to_names[gt] if gt in gene_ids_to_names else gt for gt in terms[F[gene_id,:]]]
        
        for gt in gene_terms:
            if gt not in counts_by_term:
                counts_by_term[gt] = [0, 0, 0]
            
            counts_by_term[gt][int(thebin)] += 1
    
    rows = []
    for term in counts_by_term:
        cnts = np.array(counts_by_term[term])
        normed_cnts = cnts / np.sum(cnts)

        rows.append({
            "term" : term,
            "lethal" : cnts[0],
            "sick" : cnts[1],
            "healthy" : cnts[2],
            "lethal_p" : normed_cnts[0],
            "sick_p" : normed_cnts[1],
            "healthy_p" : normed_cnts[2],
            "total" : np.sum(cnts),
            "diff" : (np.max(normed_cnts) - np.min(normed_cnts))
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, columns=['term', 'lethal', 'sick', 'healthy', "lethal_p",
         "sick_p", "healthy_p", "total", "diff"])


    # run statistical tests
    bins = ["lethal_p", "sick_p", "healthy_p"]
    n_comparisons = 0
    for i in range(len(bins)):
        for j in range(i+1, len(bins)):
            col_i = df[bins[i]]
            col_j = df[bins[j]]

            corr, rho = scipy.stats.spearmanr(col_i, col_j)
            n_comparisons += 1
            print("%s %s: %0.2f (%0.6f)" % (bins[i], bins[j], corr, rho))
    
    alpha = 0.05
    adjusted_alpha = alpha / n_comparisons
    print("Alpha = %0.6f, Adjusted Alpha = %0.6f" % (alpha, adjusted_alpha))
    
if __name__ == "__main__":
    task_file = sys.argv[1]
    go_file = sys.argv[2]
    output_path = sys.argv[3]

    main(task_file, go_file, output_path)
