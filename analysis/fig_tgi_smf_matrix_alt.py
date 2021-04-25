import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.metrics
import matplotlib.pyplot as plt 
import json
from collections import defaultdict
import matplotlib.colors
import matplotlib.ticker as ticker
import scipy.stats as stats
import numpy.random as rng 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys 
import seaborn as sns
import utils.eval_funcs as eval_funcs
BINARY_BIN_LABELS = ['Negative', 'Neutral']
BINARY_COLORS = ['magenta', 'cyan']
STAR_COLORS = ['magenta', '#007bff']
plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"

SMF_CLASSES = ['L', 'R', 'N']

plot_cfg = {
    "tick_label_size" : 40,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 82,

    "legend_size" : 42,
    "max_bars" : 4
}

MAPPING = {

    (0, 0) : 0, 
    (0, 1) : 1,
    (0, 2) : 2,
    (1, 0) : 1,
    (1, 1) : 3,
    (1, 2) : 4,
    (2, 0) : 2,
    (2, 1) : 4,
    (2, 2) : 5
}
ROW_LABELS = [
    "LL", 
    "LR",
    "LN",
    "RR",
    "RN",
    "NN"
]
ROW_ORDER = [0, 1, 2, 5, 4, 3]

def main(task_path, smf_path, output_path):

    df = pd.read_csv(task_path)
    
    smf_df = pd.read_csv(smf_path)

    smf = dict(zip(smf_df['id'], smf_df['bin']))
    
    genes_with_smf = set(smf_df['id'])

    with_smf_ix = (df['a_id'].isin(genes_with_smf)) & (df['b_id'].isin(genes_with_smf)) & (df['c_id'].isin(genes_with_smf))
    print("# pairs without smf: %d" % np.sum(~with_smf_ix))

    df = df[with_smf_ix]

    labels = BINARY_BIN_LABELS
    colors = BINARY_COLORS

    Ms = []
    for bin in range(len(labels)):
        ix = (df['bin'] == bin)
        print("# observation in %d: %d" % (bin, np.sum(ix)))

        sdf = df[ix]

        a_smf_bin = np.array([smf[i] for i in sdf['a_id']]).astype(int)
        b_smf_bin = np.array([smf[i] for i in sdf['b_id']]).astype(int)
        c_smf_bin = np.array([smf[i] for i in sdf['c_id']]).astype(int)
        
        all_bins = np.vstack((a_smf_bin, b_smf_bin, c_smf_bin)).T
        
        n_l = np.sum(all_bins == 0, axis=1)
        n_r = np.sum(all_bins == 1, axis=1)
        n_n = np.sum(all_bins == 2, axis=1)

        P = 4
        M = np.zeros((4,4,4))
        for i in range(len(a_smf_bin)):
            M[n_l[i], n_r[i], n_n[i]] += 1
        Ms.append(M)
    
    df, testr = prepare_detailed_data(Ms)
    visualize(df, testr, output_path + '_detailed.png', stars_offset=(0.2, 0.99), legend_pos=(0.53,0.91))
    df, testr = prepare_summary_data(Ms)
    visualize(df, testr, output_path + '_summary.png', stars_offset=(0.5, 0.99), legend_pos=(0.8, 0.91))

def prepare_detailed_data(Ms):
    rows = []
    vals_by_bin = []
    for bin in [0, 1]:
        M = Ms[bin]
        
        vals = []
        labels = []
        for n_n in range(M.shape[0]):
            for n_r in range(4 - n_n):
                n_l = 3 - n_n - n_r
                vals.append(M[n_l, n_r, n_n])
                rows.append({
                    "label" : "L=%d\nR=%d\nN=%d" % (n_l, n_r, n_n),
                    "bin" : BINARY_BIN_LABELS[bin], 
                    "val" : M[n_l, n_r, n_n] / np.sum(M)
                })
        vals_by_bin.append(vals)
    
    vals_by_bin = np.array(vals_by_bin)
    testr = chi2(vals_by_bin)

    return pd.DataFrame(rows), testr
    
def prepare_summary_data(Ms):
    rows = []
    vals_by_bin = []
    for bin in [0, 1]:
        M = Ms[bin]
        
        vals = []
        labels = []

        rows.append({
            "label" : "N < 2",
            "bin" : BINARY_BIN_LABELS[bin],
            "val" : np.sum(M[:, :, :2]) / np.sum(M)
        })
        vals.append(np.sum(M[:, :, :2]))

        rows.append({
            "label" : "N ≥ 2",
            "bin" : BINARY_BIN_LABELS[bin],
            "val" : np.sum(M[:, :, 2:]) / np.sum(M)
        })
        vals.append(np.sum(M[:, :, 2:]))
        
        vals_by_bin.append(vals)
    
    vals_by_bin = np.array(vals_by_bin)
    testr = chi2(vals_by_bin)

    for bin in [0, 1]:
        for lbl in [' ', '  ', '   ']:
            rows.append({
                "label" : lbl, 
                "bin" : BINARY_BIN_LABELS[bin],
                "val" : 0
            })
    
    return pd.DataFrame(rows), testr
def visualize(df, testr, output_path, stars_offset=(0.5, 0.87), legend_pos=(0.5,0.95)):
    f, ax = plt.subplots(1, 1, figsize=(15, 10))

    g = sns.barplot(x="label", 
            y="val", 
            hue="bin",
            ax=ax,
            data=df, 
            palette=['magenta', 'cyan'])

    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel("% Triplets", fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.set_xlabel("", fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(frameon=False, fontsize=plot_cfg['legend_size'], loc='center', bbox_to_anchor=legend_pos)
    plt.setp(ax.spines.values(),linewidth=plot_cfg["border_size"], color='black')
    chisq, p, ddof = testr
    stars = '*' * eval_funcs.compute_stars(p, 0.05)
    
    ax.text(stars_offset[0], stars_offset[1], stars, 
            transform=ax.transAxes,
            color=STAR_COLORS[1], ha="left", va="top", weight='bold', 
            fontsize=plot_cfg['stars_label_size'])
    

    plt.savefig(output_path, bbox_inches='tight', dpi=100)

def chi2(f_obs):

    # compute expected frequencies
    col_marginal = np.sum(f_obs, axis=1, keepdims=True)
    row_marginal = np.sum(f_obs, axis=0, keepdims=True)
    total = np.sum(row_marginal)
    f_exp = np.dot(col_marginal / total, row_marginal)
    
    chisq, p = stats.chisquare(f_obs, f_exp, axis=None)
    
    ddof = (f_obs.shape[0]-1) * (f_obs.shape[1]-1)

    return chisq, p, ddof

if __name__ == "__main__":
    task_path = sys.argv[1]
    smf_path = sys.argv[2]
    output_path = sys.argv[3]

    main(task_path, smf_path, output_path, True)
