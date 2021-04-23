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
plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"

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

def main(task_path, feature_path, ylabel, output_path, BINS=np.array(['-', 'N', '+', 'S']),
    colors = ['#FF0000', '#FFFF00', '#00CC00', '#3d77ff'],
    star_colors = ['#FF0000', 'orange', '#00CC00', '#3d77ff'] 
):

    df = pd.read_csv(task_path)
    
    

    #loader = TruePairwiseFeatureLoader(feature_path)
    loader = SumPairwiseLoader(feature_path)

    bins = sorted(np.unique(df['bin']))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    df['feature'] = loader.get_values(df)
    df['bin'] = BINS[df['bin'].astype(int)]

    ax = sns.violinplot(x="bin", y="feature", ax=ax, order=BINS, data=df, 
        palette=colors, saturation=1)
    

    
    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel(ylabel, fontsize=plot_cfg['ylabel_size'], weight='bold')
    ax.set_xlabel('')
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.xaxis.set_tick_params(length=0)
    ax.grid(False)
    plt.setp(ax.spines.values(),linewidth=plot_cfg["border_size"], color='black')

    max_val = np.max(df['feature'])
    min_val, max_val = ax.get_ylim()
    ax.set_ylim([min_val, max_val*1.3])
    ax.grid(False)
    plt.setp(ax.spines.values(),linewidth=plot_cfg["border_size"], color='black')
    
    bins = BINS
    
    # plot pvalues
    ALPHA = 0.05
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
            statistic, pvalue,_,_ = stats.median_test(a, b)
            print("%s (%0.2f) vs. %s (%0.2f): %0.6f [%0.6f]" % (bins[i], 
                np.median(a), bins[j], np.median(b), pvalue, statistic))
            
            if pvalue < adjusted_alpha:
                stars = '*' * eval_funcs.compute_stars(pvalue, adjusted_alpha)
                target_color = star_colors[j]
                ax.text(i, yoffset, stars, 
                    color=target_color, ha="center", va="center", weight='bold', 
                    fontsize=plot_cfg['stars_label_size'])
                yoffset += 0.1 * max_val

    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.show()
    
    plt.close()
    
class TruePairwiseFeatureLoader(object):

    def __init__(self, path):
        self.F = np.load(path)
    
    def get_values(self, df):
        return self.F[df['a_id'], df['b_id']]

class SumPairwiseLoader(object):

    def __init__(self, path):
        d = np.load(path)
        F = d['F']
        F = (F + d['mu']) * d['std']

        selected_feature_ix = d['feature_labels'] == "lid"
        f = F[:,selected_feature_ix]

        self.f = f
    def get_values(self, df):
        
        r = self.f[df['a_id'], :] + self.f[df['b_id'], :]

        if 'c_id' in df.columns:
            r = r + self.f[df['c_id'], :]
        
        return r 

if __name__ == "__main__":
    task_path = sys.argv[1]
    feature_path = sys.argv[2]
    output_path = sys.argv[3]

    main(task_path, feature_path, "Sum LID", output_path)
