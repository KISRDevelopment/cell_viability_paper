from pygosemsim import graph, download, annotation, term_set, similarity
import networkx as nx
import functools
import utils.yeast_name_resolver as nr 
import numpy as np
import os 
res = nr.NameResolver()

gpath = "../generated-data/ppc_yeast"
def main(term_type):
    #download.obo("go-basic")
    #download.gaf("sgd")
    
    annot = annotation.from_resource("sgd")
    
    G = graph.from_resource("go-basic")

    #print(G.nodes['GO:2001311'])
    #exit()

    similarity.precalc_lower_bounds(G)
    sf = functools.partial(term_set.sim_func, G, similarity.lin)

    sgd_to_locus = read_names("../data-sources/yeast/names.txt")

    keys = list(annot.keys())


    ppcG = nx.read_gpickle(gpath)
    nodes = sorted(ppcG.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    
    translated_keys = {}
    nodes_to_terms = {}
    for key in keys:
        if key not in sgd_to_locus:
            continue 

        locus = sgd_to_locus[key]
        unified = get_unified_name(locus)
        if unified in node_ix:
            translated_keys[key] = node_ix[unified]
            node_terms = [t for t in annot[key]["annotation"].keys() if G.nodes[t]['namespace'] == term_type]
            nodes_to_terms[key] = set(node_terms)
    
    keys = list(translated_keys.keys())
    print(len(keys))
    #print(nodes_to_terms)
    #exit()

    F = np.zeros((len(keys), len(keys)))
    for i in range(len(keys)):
        a = keys[i]
        print("%6d %s" % (i, a))

        terms_a = nodes_to_terms[a]
        if len(terms_a) == 0:
            continue 

        #print(terms_a)
        for j in range(i+1, len(keys)):
            b = keys[j]
            terms_b = nodes_to_terms[b]
            if len(terms_b) == 0:
                continue 
            
            F[translated_keys[a], translated_keys[b]] = term_set.sim_bma(terms_a, terms_b, sf)
            F[translated_keys[b], translated_keys[a]] = F[translated_keys[a], translated_keys[b]]

        print(np.min(F), np.max(F))

        output_path = "../generated-data/pairwise_features/%s_semsim_%s" % (os.path.basename(gpath), term_type)
    
        np.save(output_path, F)

def get_unified_name(locus_tag):
    locus_tag = locus_tag.lower()
    return locus_tag + '  ' + res._locus_to_alias[locus_tag][0]

def read_names(path):

    sgd_id_to_locus = {}

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            locus_tag = line[75:95].strip()
            sgd_id = line[118:129].strip()

            sgd_id_to_locus[sgd_id] = locus_tag
    
    return sgd_id_to_locus

if __name__ == "__main__":
    import sys 

    main(sys.argv[1])
