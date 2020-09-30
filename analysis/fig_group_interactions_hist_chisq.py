import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import scipy.stats
import seaborn as sns
import sklearn.metrics as metrics
import sys
import json
from scipy import interp
import os 
from seaborn.utils import remove_na
import matplotlib.patches as mpatches
import utils.eval_funcs as eval_funcs
import scipy.stats as stats
import utils.yeast_name_resolver as nr
from collections import defaultdict

res = nr.NameResolver()

plot_cfg = {
    "tick_label_size" : 45,
    "xlabel_size" : 50,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 45,
    "annot_size" : 82,
    "legend_label_size" : 42
}

ALPHA = 0.05
plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'

BIN_LABELS = ['Negative', 'Neutral', 'Positive', 'Suppression']

col_names = ['no. neg', 'no. neut', 'no. pos', 'no. sup']

COLORS = ['#FF0000', 'orange', '#00CC00', '#3d77ff']

GROUP = 'pathways'
def main(task_path, output_path):
    # genes_to_complexes = parse_yeast_complexes()
    # genes_to_pathways =  parse_kegg_pathways()
    # genes_to_cp = defaultdict(lambda: { "complexes" : set(), "pathways" : set() })

    # for k, v in genes_to_complexes.items():
    #     genes_to_cp[k]["complexes"] = v 
        
    # for k,v in genes_to_pathways.items():
    #     genes_to_cp[k]["pathways"] = v 
        
    # if GROUP == 'complexes':
    #     genes_to_groups = genes_to_complexes
    #     exclusion_criteria = lambda a, b: diff_complex_but_same_pathway(a, b, genes_to_cp)
    # else:
    #     genes_to_groups = genes_to_pathways
    #     exclusion_criteria = lambda a, b: same_complex_same_pathway(a, b, genes_to_cp)
    
    # gi_df = pd.read_csv(task_path)

    # genes_to_group = { g: list(v)[0] for g,v in genes_to_groups.items() if len(v) == 1}

    # R = count_props(gi_df, genes_to_group, exclusion_criteria)
    # np.save('../tmp/R',R)

    R = np.load('../tmp/R.npy')

    print(R)
    print(np.sum(R[:]))

    chisq, p, ddof = chi2(R)
    print("Chi2: %f" % chisq)
    print("p-value: %f" % p)
    print("ddof: %d" % ddof)

    R_normed = R / np.sum(R, axis=0, keepdims=True)
    
    df = pd.DataFrame([
        { "bin_id" : 0, "assoc" : "Within", "prop" : R_normed[0, 0] },
        { "bin_id" : 1, "assoc" : "Within", "prop" : R_normed[1, 0] },
        { "bin_id" : 2, "assoc" : "Within", "prop" : R_normed[2, 0] },
        { "bin_id" : 3, "assoc" : "Within", "prop" : R_normed[3, 0] },
        { "bin_id" : 0, "assoc" : "Across", "prop" : R_normed[0, 1] },
        { "bin_id" : 1, "assoc" : "Across", "prop" : R_normed[1, 1] },
        { "bin_id" : 2, "assoc" : "Across", "prop" : R_normed[2, 1] },
        { "bin_id" : 3, "assoc" : "Across", "prop" : R_normed[3, 1] }
    ])
    df['prop'] *= 100
    df['bin'] = [BIN_LABELS[b] for b in df['bin_id']]

    _, p_level = eval_funcs.compute_stars(p, ALPHA, return_level=True)

    f, ax = plt.subplots(1, 1, figsize=(15, 10))
    bar = sns.barplot(x="bin", 
        y="prop", 
        hue="assoc",
        data=df, 
        edgecolor='black',
        errwidth=5,
        errcolor='black',
        linewidth=plot_cfg["bar_border_size"],
        saturation=1)

    
    across_patch = mpatches.Patch(facecolor='white', fill=False, hatch='x',label='Across')
    within_patch = mpatches.Patch(facecolor='white', edgecolor='black', fill=True, label='Within')
    
    ax.legend(handles=[within_patch, across_patch], fontsize=plot_cfg['legend_label_size'])

    ylim = ax.get_ylim()
    colors = COLORS
    bin_ids = sorted(set(df['bin_id']))
    for i, b in enumerate(bar.patches):
        
        # get the hue of that bar
        hid = i // len(bin_ids)
        gid = i % len(bin_ids)

        group = bin_ids[gid]
        
        x,y = b.get_xy()
        height = b.get_height()
        width = b.get_width()

        if hid > 0:
            b.set_hatch('x')

            b.set_linewidth(0)

            b.set_facecolor(colors[group])

            b.set_alpha(0.5)
        else:
            b.set_color(colors[group])

    ax.text(0.5, 0.5, "$\chi^2_{%d}=%0.1f, \\rho < %0.4f$" % (ddof, chisq, p_level), 
        transform=ax.transAxes, fontsize=plot_cfg['legend_label_size'])

    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel("%", fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.set_xlabel("")
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=100)

def chi2(f_obs):

    # compute expected frequencies
    col_marginal = np.sum(f_obs, axis=1, keepdims=True)
    row_marginal = np.sum(f_obs, axis=0, keepdims=True)
    total = np.sum(row_marginal)
    f_exp = np.dot(col_marginal / total, row_marginal)

    print(f_obs)
    print(col_marginal)
    print(row_marginal)
    print(f_exp)

    chisq, p = stats.chisquare(f_obs, f_exp, axis=None)
    
    ddof = (f_obs.shape[0]-1) * (f_obs.shape[1]-1)

    return chisq, p, ddof

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


def count_props(gi_df, genes_to_group, exclusion_criteria):

    groups_set = set(list(genes_to_group.values()))
    groups = sorted(groups_set)
    group_ix = dict(zip(groups, range(len(groups))))

    df_a = list(gi_df['a'])
    df_b = list(gi_df['b'])
    df_bin = list(gi_df['bin'])
    
    # (bins x within/across)
    R = np.zeros((4, 2))

    n_ignored = 0
    n_no_assoc = 0
    for i in range(gi_df.shape[0]):
        a = df_a[i]
        b = df_b[i]

        if exclusion_criteria(a, b):
            n_ignored += 1
            continue

        if a in genes_to_group and b in genes_to_group:

            group_a = genes_to_group[a]
            group_b = genes_to_group[b]

            if group_a == group_b:
                R[df_bin[i], 0] += 1
            else:
                R[df_bin[i], 1] += 1
        else:
            n_no_assoc += 1

    print("Ignored: %d, no assoc: %d" % (n_ignored, n_no_assoc))
    return R

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
if __name__ == "__main__":
    main("../generated-data/task_yeast_gi_hybrid", sys.argv[1])

