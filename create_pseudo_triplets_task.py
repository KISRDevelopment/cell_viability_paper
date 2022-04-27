import numpy as np
import pandas as pd
import json
import networkx as nx 
from collections import defaultdict
import numpy.random as rng

def main(n_samples_within=5000, n_samples_across=5000):

    G = nx.read_gpickle("../generated-data/ppc_yeast")
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    with open('../generated-data/yeast_complexes.json', 'r') as f:
        genes_to_groups = json.load(f)
    
    genes_to_group = { g: list(v)[0] for g,v in genes_to_groups.items() if len(v) == 1}

    group_to_genes = defaultdict(set)
    eligible_genes = set()
    for gene, group in genes_to_group.items():
        if gene in node_ix:
            group_to_genes[group].add(gene)
            eligible_genes.add(gene)
    eligible_genes = list(eligible_genes)

    triplets_in_same_complex = set()
    groups = list(group_to_genes)
    groups = [g for g in groups if len(group_to_genes[g]) >= 3]
    while len(triplets_in_same_complex) < n_samples_within:
        # pick a random complex
        group = rng.choice(groups)
        # sample three genes
        genes = list(group_to_genes[group])
        triplet = tuple(sorted(rng.choice(genes, size=3, replace=False)))
        triplets_in_same_complex.add(triplet)

    triplets_in_diff_complexes = set()
    while len(triplets_in_diff_complexes) < n_samples_across:
        triplet = tuple(sorted(rng.choice(eligible_genes, size=3, replace=False)))
        
        a,b,c = triplet
        if (genes_to_group[a] != genes_to_group[b]) \
            or (genes_to_group[a] != genes_to_group[c]) \
            or (genes_to_group[b] != genes_to_group[c]):
            triplets_in_diff_complexes.add(triplet)

    rows = [{
        "a" : a,
        "b" : b,
        "c" : c,
        "bin" : 0,
        "a_id" : node_ix[a],
        "b_id" : node_ix[b],
        "c_id" : node_ix[c] }
        for a,b,c in triplets_in_same_complex
    ] + [ {
            "a" : a,
            "b" : b,
            "c" : c,
            "bin" : 1,
            "a_id" : node_ix[a],
            "b_id" : node_ix[b],
            "c_id" : node_ix[c] }
            for a,b,c in triplets_in_diff_complexes
    ]
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv('../generated-data/pseudo_triplets')

if __name__ == "__main__":
    main()
