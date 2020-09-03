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
BINS = np.array(['Negative', 'Neutral', 'Positive', 'Suppression'])

plot_cfg = {
    "tick_label_size" : 70,
    "xlabel_size" : 80,
    "ylabel_size" : 80,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 72,
    "annot_size" : 82,
}

def main(task_path, feature_path, ylabel, output_path):

    df = pd.read_csv(task_path)
    
    colors = ['#FF0000', 'orange', '#00CC00', '#3d77ff']
    star_colors = ['#FF0000', 'orange', '#00CC00', '#3d77ff'] 

    loader = TruePairwiseFeatureLoader(feature_path)
    
    mean_spl = loader.mean()
    print(mean_spl)
    bins = sorted(np.unique(df['bin']))

    vals = loader.get_values(df)

    val_range = np.arange(np.min(vals), np.max(vals)+1)

    M = np.zeros((len(bins), len(val_range)))
    for i in range(len(bins)):
        
        ix = df['bin'] == i
        sdf = df[ix]
        svals = vals[ix]
        
        freqs = []
        for v in val_range:
            freqs.append(np.sum(svals == v))
        
        M[i, :] = freqs 

    # normalize by row
    M = M * 100 / np.sum(M, axis=1, keepdims=True)
    max_val = 80

    fig, axes = plt.subplots(len(bins), 1, figsize=(20, 20), sharex=True)
    for i in range(len(bins)):
        bin_mean_spl = np.sum(M[i, :] * val_range / 100)
        #print(bin_mean_spl)

        axes[i].plot([mean_spl, mean_spl], [0, 100], linewidth=10, color='#525252', linestyle='--')

        axes[i].bar(val_range, M[i, :], color=colors[i])
        axes[i].yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
        axes[i].xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
        axes[i].set_xticks(val_range)
        axes[i].set_yticks([0, 25, 50, 75])
        axes[i].set_ylabel(ylabel, fontsize=plot_cfg['ylabel_size'], weight='bold')
        axes[i].yaxis.set_tick_params(length=10, width=1, which='both')
        axes[i].xaxis.set_tick_params(length=0, pad=15)
        if i == len(bins) - 1:
            axes[i].set_xlabel("Shortest Path Length", fontsize=plot_cfg['xlabel_size'], weight='bold')
        axes[i].set_ylim([0, max_val])
        axes[i].set_xlim([0, 6])
        plt.setp(axes[i].spines.values(),linewidth=plot_cfg["border_size"], color='black')
        #axes[i].set_title(BINS[i], fontsize=plot_cfg['xlabel_size'], color=colors[i])
    

    # plot pvalues
    ALPHA = 0.05
    num_comparisons = len(bins) * (len(bins)-1) / 2
    adjusted_alpha = ALPHA / num_comparisons

    for i in range(len(bins)):
        ix = df['bin'] == i
        a = vals[ix]

        a_med = np.median(a)
        yoffset = 0.95
        xoffset = 0.95
        for j in range(i+1, len(bins)):
            ix = df['bin'] == j
            b= vals[ix]

            statistic, pvalue,_,_ = stats.median_test(a, b)
            print("%s (%0.2f) vs. %s (%0.2f): %0.6f [%0.6f]" % (bins[i], 
                np.median(a), bins[j], np.median(b), pvalue, statistic))
            
            if pvalue < adjusted_alpha:
                stars = '*' * eval_funcs.compute_stars(pvalue, adjusted_alpha)
                target_color = star_colors[j]
                axes[i].text(xoffset, yoffset, stars, 
                    transform=axes[i].transAxes,
                    color=target_color, ha="right", va="top", weight='bold', 
                    fontsize=plot_cfg['stars_label_size'])
                yoffset -= 0.2

    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()
    
class TruePairwiseFeatureLoader(object):

    def __init__(self, path):
        self.F = np.load(path)
    
    def mean(self):
        indecies = np.triu_indices(self.F.shape[0], 1)

        vals = self.F[indecies]

        return np.mean(vals)
    
    def distrib(self):
        indecies = np.triu_indices(self.F.shape[0], 1)

        vals = self.F[indecies]
        print(vals.shape)
        from collections import defaultdict
        distrib = defaultdict(int)

        for v in vals:
            
            distrib[v] = distrib[v] + 1
        
        print(distrib)

    def get_values(self, df):
        return self.F[df['a_id'], df['b_id']]

if __name__ == "__main__":
    task_path = sys.argv[1]
    feature_path = sys.argv[2]
    output_path = sys.argv[3]

    main(task_path, feature_path, "%", output_path)
