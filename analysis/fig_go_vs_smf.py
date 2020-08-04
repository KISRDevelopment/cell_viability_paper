import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.metrics
import json
import sys 
import scipy.stats 
import matplotlib.colors 

plot_cfg = {
    "tick_label_size" : 28,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 24
}


def main(tbl_path, output_path):

    df = pd.read_excel(tbl_path)
    df = df.sort_values('lethal_p', ascending=False)

    vals = np.array(df[['lethal_p', 'sick_p', 'healthy_p']])
    orig_vals = vals.copy()

    min_vals = np.min(vals, axis=0, keepdims=True)
    max_vals = np.max(vals, axis=0, keepdims=True)
    vals = (vals - min_vals) / (max_vals - min_vals)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"])

    f, ax = plt.subplots(1, 1, figsize=(10, 20))
    ax.yaxis.tick_right()
    ax.imshow(vals, aspect=0.25, cmap=cmap)
    plt.setp(ax.spines.values(), linewidth=0, color='black')
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df['term'].apply(make_label), fontsize=plot_cfg['tick_label_size'])
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['L', 'R', 'N'], fontsize=plot_cfg['tick_label_size'])

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, '%0.1f' % orig_vals[i,j], 
                    color='black', ha="center", va="center", weight='bold', 
                    fontsize=plot_cfg['tick_label_size'])
    
    plt.savefig(output_path, bbox_inches='tight', dpi=100)

    #plt.show()

def make_label(s):
    return s[0].upper() + s[1:]

if __name__ == "__main__":
    tbl_path =sys.argv[1]
    main(tbl_path)
