#
#   Examples:
#       
#   - To generated LID figure:
#        python -m analysis.fig_feature_violin ../generated-data/task_yeast_smf_30 ../generated-data/features/ppc_yeast_topology.npz 11 ../tmp/fig1_e.png
# 

import seaborn as sns 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys 
import scipy.stats 
import utils.eval_funcs as eval_funcs

BINS = np.array(['L', 'R', 'N'])
ALPHA = 0.05 

plt.rcParams["font.family"] = "Liberation Serif"

plot_cfg = {
    "tick_label_size" : 50,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 82,
    "iqr_color" : "#303030",

}

def main(task_file, feature_file, fid, ylabel, output_path):
    
    # load data
    df = pd.read_csv(task_file)
    df['bin'] = BINS[df['bin'].astype(int)]

    # load LID
    d = np.load(feature_file)
    labels = d['feature_labels'].tolist()
    f = d['F'][:, fid]
    f = f * d['std'][fid] + d['mu'][fid]
    
    visualize(df, f, ylabel)
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()
    
def visualize(df, f, ylabel, ylim=None):
    df['feature'] = f[df['id']]

    print(df)
    colors = ['#FF0000', '#FFFF00', '#00CC00']
    star_colors = ['#FF0000', 'orange', '#00CC00']
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = sns.violinplot(x="bin", y="feature", ax=ax, order=BINS, data=df, 
        palette=colors, saturation=1)
    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    #ax.set_xlabel('Phenotype', fontsize=plot_cfg['xlabel_size'])
    ax.set_ylabel(ylabel, fontsize=plot_cfg['ylabel_size'], weight='bold')
    ax.set_xlabel('')
    ax.set_xlim([-0.5, 2.5])
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.xaxis.set_tick_params(length=0)

    max_val = np.max(df['feature'])
    if ylim:
        ax.set_ylim(ylim)
    else:
        min_val, max_val = ax.get_ylim()
        ax.set_ylim([min_val, max_val*1.2])
    ax.grid(False)
    plt.setp(ax.spines.values(),linewidth=plot_cfg["border_size"], color='black')
    
    bins = BINS
    
    # plot pvalues
    num_comparisons = len(bins) * (len(bins)-1) / 2
    adjusted_alpha = ALPHA / num_comparisons

    for i in range(len(bins)):
        a = df[df['bin'] == bins[i]]['feature']
        a_med = np.median(a)
        ax.plot([i, i], [a_med, a_med], 'o', color=plot_cfg['iqr_color'], markersize=15)
        iqr_lower = np.percentile(a, 25)
        iqr_upper = np.percentile(a, 75)
        ax.plot([i, i], [iqr_lower, iqr_upper], linewidth=5, color=plot_cfg['iqr_color'])
        
        yoffset = max_val
        
        for j in range(i+1, len(bins)):
            b = df[df['bin'] == bins[j]]['feature']
            statistic, pvalue = scipy.stats.kruskal(a, b)
            print("%s (%0.2f) vs. %s (%0.2f): %0.6f [%0.6f] (lens %d vs %d)" % (bins[i], 
                np.median(a), bins[j], np.median(b), pvalue, statistic, a.shape[0], b.shape[0]))
            
            if pvalue < adjusted_alpha:
                stars = '*' * eval_funcs.compute_stars(pvalue, adjusted_alpha)
                target_color = star_colors[j]
                ax.text(i, yoffset, stars, 
                    color=target_color, ha="center", va="center", weight='bold', 
                    fontsize=plot_cfg['stars_label_size'])
                yoffset += 0.1 * max_val
        

if __name__ == "__main__":
    task_file = sys.argv[1]
    feature_file = sys.argv[2]
    fid = int(sys.argv[3])
    output_path = sys.argv[4]

    main(task_file, feature_file, fid, output_path)
