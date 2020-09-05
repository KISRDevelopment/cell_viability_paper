import pandas as pd 
import numpy as np 
import sys
import utils.yeast_name_resolver as nr
from collections import defaultdict 
import json
import matplotlib.pyplot as plt 
import scipy.stats
import numpy.random as rng
import numpy.random as rng 

res = nr.NameResolver()

THRES = 0.25

BIN_LABELS = ['neg', 'neut', 'pos', 'sup']

def main(task_path, group, output_path):

    genes_to_complexes = parse_yeast_complexes()
    genes_to_pathways =  parse_kegg_pathways()
    genes_to_cp = defaultdict(lambda: { "complexes" : set(), "pathways" : set() })

    for k, v in genes_to_complexes.items():
        genes_to_cp[k]["complexes"] = v 
        
    for k,v in genes_to_pathways.items():
        genes_to_cp[k]["pathways"] = v 
        

    if group == 'complexes':
        genes_to_groups = genes_to_complexes
        exclusion_criteria = lambda a, b: diff_complex_but_same_pathway(a, b, genes_to_cp)
    else:
        genes_to_groups = genes_to_pathways
        exclusion_criteria = lambda a, b: same_complex_same_pathway(a, b, genes_to_cp)
    all_groups = []
    for k, v in genes_to_groups.items():
        all_groups.extend(v)
    
    summary_df_rows = []

    n_groups = len(set(all_groups))
    summary_df_rows.append(("No. Groups", n_groups))
    
    n_assoc = len(genes_to_groups)
    summary_df_rows.append(("No. Genes associated with a group", n_assoc))
    
    genes_to_group = { g: list(v)[0] for g,v in genes_to_groups.items() if len(v) == 1}

    n_one_pathway = len(genes_to_group)
    summary_df_rows.append(("No. Genes associated with only one group", n_one_pathway))
    
    n_multiple_pathways = n_assoc - n_one_pathway
    summary_df_rows.append(("No. Genes removed due to involvement in multiple groups", n_multiple_pathways))
    
    print(summary_df_rows)

    gi_df = pd.read_csv(task_path)

    all_groups = []
    for g,pl in genes_to_group.items():
        all_groups.append(pl)
    all_groups = set(all_groups)

    R, group_ix, examined_genes = count_props(gi_df, genes_to_group, exclusion_criteria)
    
    writer = pd.ExcelWriter(output_path)

    raw_df, _, _ = summarize_groups(R, group_ix, genes_to_group, examined_genes, normalize=False)
    normed_df, unfiltered_df, columns = summarize_groups(R, group_ix, genes_to_group, examined_genes, normalize=True)


    summary_df_rows.append(("No. Groups that involve genes associated with one pathway", unfiltered_df.shape[0]))
    summary_df_rows.append(("No. Groups removed because they don't meet filtering criteria", (unfiltered_df.shape[0] - normed_df.shape[0])))
    summary_df_rows.append(("Final No. Groups", normed_df.shape[0]))
    summary_df = pd.DataFrame(summary_df_rows)

    unfiltered_df.to_excel(writer, sheet_name='unfiltered', columns=columns, index=False)

    raw_df.to_excel(writer, sheet_name='counts', columns=columns, index=False)

    summary_df.to_excel(writer, sheet_name='normalized', index=False, header=False)
    normed_df.to_excel(writer, sheet_name='normalized', startrow=summary_df.shape[0]+1, columns=columns, index=False)

    writer.save()

    # print("Paired t-test,,p-value")
    # for b in range(len(BIN_LABELS)):
    #     within_p = normed_df["no. %s within" % (BIN_LABELS[b])]
    #     across_p = normed_df["no. %s across" % (BIN_LABELS[b])]
        
    #     statistic, pvalue = scipy.stats.ttest_rel(within_p, across_p)
    #     print("%s,,%f" % (BIN_LABELS[b], pvalue))
        
    # ix_group = dict(zip(group_ix.values(), group_ix.keys()))
    # target_group_ix = [group_ix[g] for g in normed_df['group']]

    # rng.shuffle(target_group_ix)

    # R = R[target_group_ix, :, :]
    # R = R[:, target_group_ix, :]

    # output_path = '../visualization/complexes_pathways/R_%s' % GROUP
    # if RANDOMIZE:
    #     output_path = '../visualization/complexes_pathways/R_%s_random' % GROUP
    # np.savez(output_path , R=R, groups=[ix_group[i] for i in target_group_ix])

def diff_complex_but_same_pathway(a, b, genes_to_cp):

    # is same pathway?
    intersect_pathways = genes_to_cp[a]["pathways"].intersection(genes_to_cp[b]["pathways"])
    if len(intersect_pathways) == 0:
        return False 
    
    intersect_complexes = genes_to_cp[a]["complexes"].intersection(genes_to_cp[b]["complexes"])
    # different complexes
    return len(intersect_complexes) == 0

def same_complex_same_pathway(a, b, genes_to_cp):

    intersect_pathways = genes_to_cp[a]["pathways"].intersection(genes_to_cp[b]["pathways"])
    if len(intersect_pathways) == 0:
        return False 
    
    intersect_complexes = genes_to_cp[a]["complexes"].intersection(genes_to_cp[b]["complexes"])
    r = len(intersect_complexes) > 0

    return r 
def summarize_groups(R, group_ix, genes_to_group, examined_genes, normalize=False):

    R = R.copy()

    rows = []
    R_interactions = R[:, :, [0,2,3]]
    n_skipped = 0
    
    n_genes = len(genes_to_group)
    

    for group, gid in group_ix.items():
        n_total = np.sum(R[gid, :, :])
        n_within = np.sum(R[gid, gid, :])
        # if n_within == 0 or n_total == 0:
        #     n_skipped += 1
        #     #print("Skipping %s: %d %d" % (group, n_within, n_total)) 
        #     continue 
        
        within_distrib = R[gid, gid, :]
        if normalize:
            within_distrib /= n_within
        
        across = R[gid, np.arange(R.shape[0]) != gid, :]
        n_across = np.sum(across)
        across_distrib = np.sum(across, axis=0)
        if normalize:
            across_distrib /= n_across

        interactions = R_interactions[gid, :, :]


        n_group_size = len([g for g in genes_to_group if genes_to_group[g] == group])

        n_examined_genes = len(examined_genes[group]["from"])
        n_within_examined = len(examined_genes[group]["within"])
        n_across_examined = len(examined_genes[group]["across"])
        
        row = {
            "group" : group,
            "size" : n_group_size,
            "no. genes - exams" : n_examined_genes,
            "no. genes - exams within" : n_within_examined,
            "no. genes - exams across" : n_across_examined,
            "no. exams" : n_total,
            "no. exams within" : n_within,
            "no. exams across" : n_across,
            "no. interactions" : np.sum(interactions),
            "no. within interactions" : np.sum(interactions[gid, :]),
            "no. across interactions" : np.sum(interactions[np.arange(R.shape[0]) != gid, :])
        }

        columns = ["group", "size", "no. genes - exams", "no. genes - exams within", "no. genes - exams across", "no. exams", "no. exams within", "no. exams across", "no. interactions", "no. within interactions", "no. across interactions"]
        for b in range(R.shape[2]):
            row["no. %s within" % (BIN_LABELS[b])] = within_distrib[b]
            row["no. %s across" % (BIN_LABELS[b])] = across_distrib[b]
            columns.extend(["no. %s within" % (BIN_LABELS[b]), "no. %s across" % (BIN_LABELS[b])])
        
        row["JS Dst"] = jensen_shannon_distance(within_distrib, across_distrib)
        columns.append("JS Dst")

        rows.append(row)
    
    df = pd.DataFrame(rows)
    unfiltered_df = df.copy()

    # apply filter criteria
    within_filter = df['no. genes - exams within'] >= THRES * df['size']
    across_filter = df['no. genes - exams across'] >= THRES * n_genes
    df = df[within_filter & across_filter]

    return df, unfiltered_df, columns

def count_props(gi_df, genes_to_group, exclusion_criteria):

    groups_set = set(list(genes_to_group.values()))
    groups = sorted(groups_set)
    group_ix = dict(zip(groups, range(len(groups))))

    df_a = list(gi_df['a'])
    df_b = list(gi_df['b'])
    df_bin = list(gi_df['bin'])
    
    R = np.zeros((len(groups), len(groups), 4))


    examined_genes = defaultdict(lambda: { "within" : set(), "across" : set(), "from" : set() })
    n_ignored = 0
    n_interactions = 0
    for i in range(gi_df.shape[0]):
        a = df_a[i]
        b = df_b[i]

        # exclude genes pairs in same pathway but different complex
        if exclusion_criteria(a, b):
            n_ignored += 1
            continue
        
        if a in genes_to_group and b in genes_to_group:
            
            if df_bin[i] != 1:
                n_interactions += 1

            group_a = genes_to_group[a]
            group_b = genes_to_group[b]

            group_a_ix = group_ix[group_a]
            group_b_ix = group_ix[group_b]

            R[group_a_ix, group_b_ix, df_bin[i]] += 1
            R[group_b_ix, group_a_ix, df_bin[i]] += 1

            if group_a == group_b:
                examined_genes[group_a]["within"].add(a)
                examined_genes[group_a]["within"].add(b)
            else:
                examined_genes[group_a]["across"].add(b)
                examined_genes[group_b]["across"].add(a)

            examined_genes[group_a]["from"].add(a)
            examined_genes[group_b]["from"].add(b)

    print("# interactions: %d" % n_interactions)
    print("Ignored %d in exclusion list" % n_ignored)
    
    return R, group_ix, examined_genes
    

def parse_kegg_pathways():

    with open('../data-sources/yeast/kegg_pathways', 'r') as f:
        genes_to_pathways = json.load(f)
    
    with open('../data-sources/yeast/kegg_names.json', 'r') as f:
        kegg_names = json.load(f)
    
    for k in genes_to_pathways.keys():
        pnames = [kegg_names[p] for p in genes_to_pathways[k]]
        genes_to_pathways[k] = pnames 

    genes_to_pathways = {res.get_unified_name(g) : set(genes_to_pathways[g]) for g in genes_to_pathways}

    return genes_to_pathways

def parse_yeast_complexes():

    df = pd.read_excel('../data-sources/yeast/CYC2008_complex.xls')

    df['gene'] = [res.get_unified_name(g.lower()) for g in df['ORF']]

    df_gene = list(df['gene'])
    df_complex = list(df['Complex'])

    genes_to_complexes = defaultdict(set)
    for i in range(df.shape[0]):
        g = df_gene[i]
        genes_to_complexes[g].add(df_complex[i])
    
    
    return genes_to_complexes

# https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d
def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance
if __name__ == "__main__":
    
    main('../generated-data/task_yeast_gi_hybrid', 'complexes', '../generated-data/complexes.xlsx')
