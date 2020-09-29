import numpy as np 
import pandas as pd 
import sys 
import networkx as nx 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats
import matplotlib.colors
from collections import defaultdict
import json
import utils.yeast_name_resolver as nr
res = nr.NameResolver()

plot_cfg = {
    "tick_label_size" : 50,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 82,
    "legend_size" : 38
}
plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'
def main(gpath, sgo_path):

    genes_to_complexes = parse_yeast_complexes()
    genes_to_pathways =  parse_kegg_pathways()
    genes_to_cp = defaultdict(lambda: { "complexes" : set(), "pathways" : set() })

    for k, v in genes_to_complexes.items():
        genes_to_cp[k]["complexes"] = v 
        
    for k,v in genes_to_pathways.items():
        genes_to_cp[k]["pathways"] = v 

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())

    d = np.load(sgo_path)
    F = d['F']
    labels = d['feature_labels']

    ix_in_complex = np.array([len(genes_to_cp[g]['complexes']) > 0 for g in nodes])
    ix_in_pathway = np.array([len(genes_to_cp[g]['pathways']) > 0 for g in nodes])

    ix_no_sgo = np.sum(F, axis=1) == 0

    print("Number of genes without SGO: %d (%0.2f)" % (np.sum(ix_no_sgo), np.mean(ix_no_sgo)))

    f_obs_complex = group_analysis(ix_in_complex, ~ix_no_sgo)
    f_obs_pathway = group_analysis(ix_in_pathway, ~ix_no_sgo)
    
    
    df = pd.DataFrame([
        { "type" : "Complex", "has_sgo" : "No sGO", "prop" : f_obs_complex[1, 0] / np.sum(ix_no_sgo) },
        { "type" : "Complex", "has_sgo" : "Has sGO", "prop" : f_obs_complex[1, 1] / np.sum(~ix_no_sgo) },
        { "type" : "Pathway", "has_sgo" : "No sGO", "prop" : f_obs_pathway[1, 0] / np.sum(ix_no_sgo) },
        { "type" : "Pathway", "has_sgo" : "Has sGO", "prop" : f_obs_pathway[1, 1] / np.sum(~ix_no_sgo) },  
    ])
    df['prop'] *= 100

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    bar = sns.barplot(x="has_sgo", y="prop", hue="type",
        data=df, ax=ax, saturation=1)    
    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel('% of Genes in Group', fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlabel('Group', fontsize=plot_cfg['xlabel_size'], fontweight='bold')
    ax.legend(fontsize=plot_cfg['legend_size'])

    plt.savefig('../figures/no_sgo_complex_pathway_membership.png', bbox_inches='tight', dpi=100)


def group_analysis(ix_group, ix_sgo):

    # (with group x with sgo)
    f_obs = np.zeros((2, 2))

    f_obs[0, 0] = np.sum(~ix_group & ~ix_sgo)
    f_obs[0, 1] = np.sum(~ix_group & ix_sgo)
    f_obs[1, 0] = np.sum(ix_group & ~ix_sgo)
    f_obs[1, 1] = np.sum(ix_group & ix_sgo)

    # compute expected frequencies
    col_marginal = np.sum(f_obs, axis=1, keepdims=True) # (2x1)
    row_marginal = np.sum(f_obs, axis=0, keepdims=True) # (1x2)
    total = np.sum(row_marginal)
    f_exp = np.dot(col_marginal / total, row_marginal)

    print(f_obs)
    print(col_marginal)
    print(row_marginal)

    print(f_exp)

    chisq, p = stats.chisquare(f_obs, f_exp, axis=None)
    print("Chi2 Statistic: %f, p: %f" % (chisq, p))

    return f_obs

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
    
    main('../generated-data/ppc_yeast', '../generated-data/features/ppc_yeast_common_sgo.npz')

