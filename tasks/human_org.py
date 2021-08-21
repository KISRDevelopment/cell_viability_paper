import os 
import sys 
import pandas as pd 
import numpy as np 
import networkx as nx 

def main(gpath, cell_smf_task_path, output_path, use_haploinsufficient=False):

    task_df = pd.read_csv(cell_smf_task_path)
    ix = task_df['bin'] == 0
    lethal_genes = set(task_df[ix]['gene'])
    ix = task_df['bin'] == 1
    sick_genes = set(task_df[ix]['gene'])
    ix = task_df['bin'] == 2
    viable_cell_genes = set(task_df[ix]['gene'])

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    file = '../data-sources/human/gnomad.v2.1.1.lof_metrics.by_gene.txt'

    df = pd.read_csv(file, sep='\t')
    df['gene'] = df['gene'].str.lower()
    df = df[df['gene'].isin(node_ix)]

    ix = (df['obs_lof'] == 0) & (df['obs_hom_lof'] == 0) & (df['obs_het_lof'] == 0)
    lethal_set1 = set(df[ix]['gene'])

    ix = (df['obs_mis'] > 1) & (df['obs_hom_lof'] == 0) & (df['obs_het_lof'] > 1)
    lethal_set2 = set(df[ix]['gene'])

    viable_df = pd.read_csv("../data-sources/human/supplementary-dataset-7-hom-ko-genes.txt", header=None)
    viable_genes = set(viable_df[0].str.lower()).intersection(set(nodes))

    print("Lethal set 1: %d, Lethal set 2: %d, Viables: %d" % (len(lethal_set1), len(lethal_set2), len(viable_genes)))


    lethal_set1 = lethal_set1 - lethal_genes - sick_genes
    lethal_set2 = lethal_set2 - lethal_genes - sick_genes
    viable_genes = viable_genes - lethal_genes - sick_genes

    print("After filtering out cell lethal and sick:")
    print("Lethal set 1: %d, Lethal set 2: %d, Viables: %d" % (len(lethal_set1), len(lethal_set2), len(viable_genes)))

    print("Common in set 1 and 2: %d" % (len(lethal_set1.intersection(lethal_set2))))
    print("Common in set 1 and 3: %d" % (len(lethal_set1.intersection(viable_genes))))
    print("Common in set 2 and 3: %d" % (len(lethal_set2.intersection(viable_genes))))

    lethal_set1 = lethal_set1 - viable_genes - lethal_set2
    lethal_set2 = lethal_set2 - viable_genes

    print("After filtering out viables:")
    print("Lethal set 1: %d, Lethal set 2: %d, Viables: %d" % (len(lethal_set1), len(lethal_set2), len(viable_genes)))

    print("Common in set 1 and 2: %d" % (len(lethal_set1.intersection(lethal_set2))))
    print("Common in set 1 and 3: %d" % (len(lethal_set1.intersection(viable_genes))))
    print("Common in set 2 and 3: %d" % (len(lethal_set2.intersection(viable_genes))))


    # with pd.ExcelWriter('../tmp/human_org_data.xlsx') as writer:
    #     pd.DataFrame({ "gene" : list(lethal_set1) }).to_excel(writer, sheet_name="Haploinsufficient", index=False)
    #     pd.DataFrame({ "gene" : list(lethal_set2) }).to_excel(writer, sheet_name="Lethal", index=False)
    #     pd.DataFrame({ "gene" : list(viable_genes) }).to_excel(writer, sheet_name="Viables", index=False)

    definitive_lethal_set = lethal_set1 if use_haploinsufficient else lethal_set2
    lethal_rows = [{ "gene" : g, "bin" : 0, "cs" : 0, "std" : 0 } for g in definitive_lethal_set]
    all_genes = set(nodes)
    viable_genes_all = all_genes - lethal_set1 - lethal_set2 - lethal_genes
    viable_rows = [{ "gene" : g, "bin" : 1, "cs" : 1, "std" : 0 } for g in viable_genes_all]
    rows = lethal_rows + viable_rows
    org_task_df = pd.DataFrame(rows)
    org_task_df['id'] = [node_ix[r['gene']] for r in rows]
    print([np.sum(org_task_df['bin'] == b) for b in [0,1]])

    #definitive_lethal_set = lethal_set1 if use_haploinsufficient else lethal_set2
    # his_rows = [{ "gene" : g, "bin" : 0, "cs" : 0, "std" : 0 } for g in lethal_set1]
    # lethal_rows = [{ "gene" : g, "bin" : 1, "cs" : 0, "std" : 0 } for g in lethal_set2]
    # all_genes = set(nodes)
    # viable_genes_all = all_genes - lethal_set1 - lethal_set2 - lethal_genes - sick_genes
    # viable_rows = [{ "gene" : g, "bin" : 2, "cs" : 1, "std" : 0 } for g in viable_genes_all]
    # rows = his_rows + lethal_rows + viable_rows
    # org_task_df = pd.DataFrame(rows)
    # org_task_df['id'] = [node_ix[r['gene']] for r in rows]

    # print([np.sum(org_task_df['bin'] == b) for b in [0,1, 2]])

    #org_task_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    gpath = sys.argv[1]
    cell_smf_task_path = sys.argv[2]
    output_path = sys.argv[3]
    main(gpath, cell_smf_task_path, output_path)