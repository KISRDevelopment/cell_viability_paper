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

BINS = np.array(['L', 'R', 'N'])

plt.rcParams["font.family"] = "Liberation Serif"


def main(feature_file, fid, output_path, remove_zeros=False):
    
    d = np.load(feature_file)
    labels = d['feature_labels'].tolist()
    f = d['F'][:, fid]
    f = f * d['std'][fid] + d['mu'][fid]
    
    if remove_zeros:
        ix = f > 0
        f = f[ix]
    
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.distplot(a=f, norm_hist=True, kde=False, ax=ax)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_xlabel(labels[fid], fontsize=22)
    ax.set_ylabel('Frequency', fontsize=22)

    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.xaxis.set_tick_params(length=0)

    ax.grid(False)
    plt.setp(ax.spines.values(), linewidth=2, color='black')

    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()

if __name__ == "__main__":
    feature_file = sys.argv[1]
    fid = int(sys.argv[2])
    output_path = sys.argv[3]

    main(feature_file, fid, output_path)
