import numpy as np 
import networkx as nx 
import os 
import sys 
from pygosemsim import graph, download, annotation, term_set
import pandas as pd 
import feature_preprocessing.pairwise_go_semsim
import utils.yeast_name_resolver as nr
myers2006path = "../data-sources/myers2006.csv"

res = nr.NameResolver()

gpath = "../generated-data/ppc_yeast"
def main():
    download.obo("go-basic")
    download.gaf("sgd")

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    G = graph.from_resource("go-basic")

    df = pd.read_csv(myers2006path, sep="\t")

    df['namespace'] = [G.nodes[n]['namespace'] if n in G else None for n in df['GO ID']]
    print(df)

    ix = (df['namespace'] == 'biological_process') & (df['# of S. cerevisiae annotations (direct and indirect)'] > 3)
    df = df[ix]
    
    eligible_terms = set(df['GO ID'])
    print("%d eligible terms" % len(eligible_terms))

    sgd_to_locus = feature_preprocessing.pairwise_go_semsim.read_names("../data-sources/yeast/names.txt")


    annot = annotation.from_resource("sgd")
    keys = list(annot.keys())

    translated_keys = {}
    nodes_to_terms = {}
    for key in keys:
        if key not in sgd_to_locus:
            continue 

        locus = sgd_to_locus[key]
        unified = get_unified_name(locus)
        if unified in node_ix:
            translated_keys[key] = node_ix[unified]
            node_terms = [t for t in annot[key]["annotation"].keys()]
            nodes_to_terms[key] = set(node_terms) & eligible_terms

    keys = list(translated_keys.keys())
    print(len(keys))

    #print(np.max([len(s) for s in nodes_to_terms.values()]))
    F = np.zeros((len(keys), len(keys)))
    for i in range(len(keys)):
        a = keys[i]
        print("%6d %s" % (i, a))

        terms_a = nodes_to_terms[a]
        if len(terms_a) == 0:
            continue 

        for j in range(i+1, len(keys)):
            b = keys[j]
            terms_b = nodes_to_terms[b]
            if len(terms_b) == 0:
                continue 
            
            F[translated_keys[a], translated_keys[b]] = len(terms_a & terms_b)
            F[translated_keys[b], translated_keys[a]] = F[translated_keys[a], translated_keys[b]]

        print(np.min(F), np.max(F))

    output_path = "../generated-data/pairwise_features/%s_common_functions" % (os.path.basename(gpath))

    np.save(output_path, F)
def get_unified_name(locus_tag):
    locus_tag = locus_tag.lower()
    return locus_tag + '  ' + res._locus_to_alias[locus_tag][0]

if __name__ == "__main__":
    main()
