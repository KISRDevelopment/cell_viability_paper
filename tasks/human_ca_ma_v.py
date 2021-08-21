import os 
import sys 
import pandas as pd 
import numpy as np 
import networkx as nx 

def main(gpath, cell_smf_task_path, output_path, use_haploinsufficient=False):

    task_df = pd.read_csv(cell_smf_task_path)
    ix = task_df['bin'] == 0
    cell_lethal_genes = set(task_df[ix]['gene'])
    ix = task_df['bin'] == 1
    cell_sick_genes = set(task_df[ix]['gene'])
    ix = task_df['bin'] == 2
    cell_viable_genes = set(task_df[ix]['gene'])
    cell_genes = set(task_df['gene'])
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))


    file = '../data-sources/human/gnomad.v2.1.1.lof_metrics.by_gene.txt'

    df = pd.read_csv(file, sep='\t')
    df['gene'] = df['gene'].str.lower()
    df = df[df['gene'].isin(node_ix)]

    ix = (df['obs_mis'] > 1) & (df['obs_hom_lof'] == 0) & (df['obs_het_lof'] > 1)
    candidate_ma_genes = set(df[ix]['gene'])

    viable_df = pd.read_csv("../data-sources/human/supplementary-dataset-7-hom-ko-genes.txt", header=None)
    viable_genes = set(viable_df[0].str.lower()).intersection(set(nodes))
    print("Viables that are CA Lethal: %d" % len(viable_genes.intersection(cell_lethal_genes)))
    
    print("Candidate MA Genes: %d, Viables: %d" % (len(candidate_ma_genes), len(viable_genes)))

    final_ma_genes = candidate_ma_genes - cell_lethal_genes - cell_sick_genes - viable_genes
    print("Final MA Genes after subtracting CA and RG and  multi-cellular Viables: %d" % (len(final_ma_genes)))

    #viable_genes = (cell_viable_genes  viable_genes) - final_ma_genes - cell_lethal_genes
    #print("Final viables: %d" % len(viable_genes))

    #cell_viable_genes = cell_viable_genes - final_ma_genes - cell_lethal_genes
    #mc_viable_genes = viable_genes - final_ma_genes - cell_lethal_genes

    # assert len(cell_lethal_genes.intersection(final_ma_genes)) == 0
    # assert len(cell_lethal_genes.intersection(cell_viable_genes)) == 0
    # assert len(cell_lethal_genes.intersection(mc_viable_genes)) == 0

    # print("Cell Viable: %d" % len(cell_viable_genes - final_ma_genes - cell_lethal_genes))
    # gene_sets = [cell_lethal_genes, final_ma_genes, mc_viable_genes]
    # rows = []
    # for i, s in enumerate(gene_sets):
    #     rows.extend([
    #         { "gene" : g, "bin" : i } for g in s
    #     ])
    
    # org_task_df = pd.DataFrame(rows)
    # org_task_df['id'] = [node_ix[r['gene']] for r in rows]

    # print([np.sum(org_task_df['bin'] == b) for b in [0,1, 2, 3]])

    # org_task_df.to_csv(output_path, index=False)

    # l_l = cell_lethal_genes.intersection(candidate_ma_genes)
    # l_v = cell_lethal_genes.intersection(viable_genes)
    # l_ur = cell_lethal_genes - (candidate_ma_genes | viable_genes)
    # v_l = (cell_viable_genes | cell_sick_genes).intersection(candidate_ma_genes)
    # v_v = (cell_viable_genes | cell_sick_genes).intersection(viable_genes)
    # v_ur = (cell_viable_genes | cell_sick_genes) - (candidate_ma_genes | viable_genes)
    # ur_l = candidate_ma_genes - cell_genes
    # ur_v = viable_genes - cell_genes 

    # sets = [l_l, l_v, l_ur, v_l, v_v, v_ur, ur_l, ur_v]
    # labels = ['Lethal (Cellular) -> Lethal (Multicellular)', 
    #           'Lethal (Cellular) -> Viable (Multicellular)',
    #           'Lethal (Cellular) -> Unreported (Multicellular)',
    #           'Viable (Cellular) -> Lethal (Multicellular)', 
    #           'Viable (Cellular) -> Viable (Multicellular)', 
    #           'Viable (Cellular) -> Unreported (Multicellular)',
    #           'Unreported (Cellular) -> Lethal (Multicellular)',
    #           'Unreported (Cellular) -> Viable (Multicellular)'
    # ]
    # for l, s in zip(labels, sets):
    #     f,t = l.split(' -> ')
    #     print("%32s %32s %8d" % (f, t, len(s)))

    from_sets = [cell_lethal_genes, cell_sick_genes, cell_viable_genes]
    from_labels = ['Lethal (Cellular)', 'Sick (Cellular)', 'Viable (Cellular)']
    to_sets = [candidate_ma_genes, viable_genes]
    to_labels = ['Lethal (Multicellular)', 'Viable (Multicellular)']
    
    for from_lbl, from_set in zip(from_labels, from_sets):
        unreported = len(from_set)
        for to_lbl, to_set in zip(to_labels, to_sets):
            intersection = from_set & to_set 
            print("%32s %32s %8d %8.2f%%" % (from_lbl, to_lbl, len(intersection), len(intersection) * 100 / len(from_set)))
            unreported -= len(intersection)
        print("%32s %32s %8d %8.2f%%" % (from_lbl, 'Unreported', unreported, unreported * 100 / len(from_set)))
        print()

if __name__ == "__main__":
    gpath = sys.argv[1]
    cell_smf_task_path = sys.argv[2]
    output_path = sys.argv[3]
    main(gpath, cell_smf_task_path, output_path)