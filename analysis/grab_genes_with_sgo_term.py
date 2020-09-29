import numpy as np 
import networkx as nx 
import sys 
import os 
import pandas as pd 
import json 

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    goids_to_names = json.load(f)

def main():
    # sgo_path = "../generated-data/features/ppc_human_common_sgo.npz"
    # gpath = '../generated-data/ppc_human'
    # d = np.load(sgo_path)
    # F = d['F']
    # labels = np.array([goids_to_names[l] for l in d['feature_labels']])

    # human_cell_lethal_both_ids = set(get_genes(gpath, sgo_path, 
    #      "../generated-data/task_human_cell_org_lethal", 0, [31, 42, 19, 43, 36, 11]))
    
    # human_cell_lethal_specific_ids = set(get_genes(gpath, sgo_path, 
    #      "../generated-data/task_human_cell_org_lethal", 0, [3, 37, 4, 44, 6])) 

    # G = nx.read_gpickle(gpath)
    # nodes = sorted(G.nodes())

    # overlap = human_cell_lethal_both_ids.intersection(human_cell_lethal_specific_ids)
    
    # print("Human Cell Lethal with sGO Terms that are Cell-Lethal in Both Organisms")
    # for idx in human_cell_lethal_both_ids-overlap:
    #     terms = labels[F[idx,:].astype(bool)]
    #     print("%s\t%s" % (nodes[idx], ", ".join(terms)))
    # print("\nHuman Cell Lethal with sGO Terms that are Cell-Lethal in Human Only")
    # for idx in human_cell_lethal_specific_ids-overlap:
    #     terms = labels[F[idx,:].astype(bool)]
    #     print("%s\t%s" % (nodes[idx], ", ".join(terms)))
    # print("\nOverlapping Genes between Two Sets")
    # for idx in overlap:
    #     terms = labels[F[idx,:].astype(bool)]
    #     print("%s\t%s" % (nodes[idx], ", ".join(terms)))

    # sgo_path = "../generated-data/features/ppc_human_common_sgo.npz"
    # gpath = '../generated-data/ppc_human'
    # d = np.load(sgo_path)
    # F = d['F']
    # labels = np.array([goids_to_names[l] for l in d['feature_labels']])

    # human_cell_lethal_both_ids = set(get_genes(gpath, sgo_path, 
    #      "../generated-data/task_human_cell_org_lethal", 1, [16, 0, 40, 1, 23]))
    
    # human_cell_lethal_specific_ids = set(get_genes(gpath, sgo_path, 
    #      "../generated-data/task_human_cell_org_lethal", 1, [33])) 

    # G = nx.read_gpickle(gpath)
    # nodes = sorted(G.nodes())

    # overlap = human_cell_lethal_both_ids.intersection(human_cell_lethal_specific_ids)
    
    # print("Human Organismal Lethal with sGO Terms that are Organismal-Lethal in Both Organisms")
    # for idx in human_cell_lethal_both_ids-overlap:
    #     terms = labels[F[idx,:].astype(bool)]
    #     print("%s\t%s" % (nodes[idx], ", ".join(terms)))
    # print("\nHuman Organismal Lethal with sGO Terms that are Organismal-Lethal in Human Only")
    # for idx in human_cell_lethal_specific_ids-overlap:
    #     terms = labels[F[idx,:].astype(bool)]
    #     print("%s\t%s" % (nodes[idx], ", ".join(terms)))
    # print("\nOverlapping Genes between Two Sets")
    # for idx in overlap:
    #     terms = labels[F[idx,:].astype(bool)]
    #     print("%s\t%s" % (nodes[idx], ", ".join(terms)))

    # sgo_path = "../generated-data/features/ppc_dro_common_sgo.npz"
    # gpath = '../generated-data/ppc_dro'
    # d = np.load(sgo_path)
    # F = d['F']
    # labels = np.array([goids_to_names[l] for l in d['feature_labels']])

    # human_cell_lethal_both_ids = set(get_genes(gpath, sgo_path, 
    #      "../generated-data/task_dro_cell_org_lethal", 0, [31, 42, 19, 43, 36, 11]))
    
    # human_cell_lethal_specific_ids = set(get_genes(gpath, sgo_path, 
    #      "../generated-data/task_dro_cell_org_lethal", 0, [29, 21, 28, 32, 39, 33])) 

    # G = nx.read_gpickle(gpath)
    # nodes = sorted(G.nodes())

    # overlap = human_cell_lethal_both_ids.intersection(human_cell_lethal_specific_ids)
    
    # print("Dmel Cell Lethal with sGO Terms that are Cell-Lethal in Both Organisms")
    # for idx in human_cell_lethal_both_ids-overlap:
    #     terms = labels[F[idx,:].astype(bool)]
    #     print("%s\t%s" % (nodes[idx], ", ".join(terms)))
    # print("\nDmel Cell Lethal with sGO Terms that are Cell-Lethal in Dmel Only")
    # for idx in human_cell_lethal_specific_ids-overlap:
    #     terms = labels[F[idx,:].astype(bool)]
    #     print("%s\t%s" % (nodes[idx], ", ".join(terms)))
    # print("\nOverlapping Genes between Two Sets")
    # for idx in overlap:
    #     terms = labels[F[idx,:].astype(bool)]
    #     print("%s\t%s" % (nodes[idx], ", ".join(terms)))

    sgo_path = "../generated-data/features/ppc_dro_common_sgo.npz"
    gpath = '../generated-data/ppc_dro'
    d = np.load(sgo_path)
    F = d['F']
    labels = np.array([goids_to_names[l] for l in d['feature_labels']])

    human_cell_lethal_both_ids = set(get_genes(gpath, sgo_path, 
         "../generated-data/task_dro_cell_org_lethal", 1, [16, 0, 40, 1, 23]))
    
    human_cell_lethal_specific_ids = set(get_genes(gpath, sgo_path, 
         "../generated-data/task_dro_cell_org_lethal", 1, [37, 4, 8, 2, 22])) 

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())

    overlap = human_cell_lethal_both_ids.intersection(human_cell_lethal_specific_ids)
    
    print("Dmel Organismal Lethal with sGO Terms that are Organismal-Lethal in Both Organisms")
    for idx in human_cell_lethal_both_ids-overlap:
        terms = labels[F[idx,:].astype(bool)]
        print("%s\t%s" % (nodes[idx], ", ".join(terms)))
    print("\nDmel Organismal Lethal with sGO Terms that are Organismal-Lethal in Dmel Only")
    for idx in human_cell_lethal_specific_ids-overlap:
        terms = labels[F[idx,:].astype(bool)]
        print("%s\t%s" % (nodes[idx], ", ".join(terms)))
    print("\nOverlapping Genes between Two Sets")
    for idx in overlap:
        terms = labels[F[idx,:].astype(bool)]
        print("%s\t%s" % (nodes[idx], ", ".join(terms)))

def get_genes(gpath, sgo_path, task_path, target_bin, term_ids):

    d = np.load(sgo_path)
    F = d['F']
    labels = np.array([goids_to_names[l] for l in d['feature_labels']])

    subF = F[:, term_ids]
    genes_with_terms_ix = np.sum(subF, axis=1) > 0

    df = pd.read_csv(task_path)
    df = df[df['bin'] == target_bin]
    genes_in_bin_ix = np.zeros(F.shape[0]).astype(bool)
    genes_in_bin_ix[df['id']] = 1

    eligible_genes_ix = genes_with_terms_ix & genes_in_bin_ix

    return np.where(eligible_genes_ix)[0]
    
if __name__ == "__main__":
    main()