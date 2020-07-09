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


def main():
    task_file = sys.argv[1]
    feature_file = sys.argv[2]
    fid = int(sys.argv[3])
    output_path = sys.argv[4]

    # load data
    df = pd.read_csv(task_file)
    df['bin'] = BINS[df['bin'].astype(int)]

    # load LID
    d = np.load(feature_file)
    labels = d['feature_labels'].tolist()
    f = d['F'][:, fid]
    f = f * d['std'][fid] + d['mu'][fid]
    
    visualize(df, f, labels[fid])
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()
    
def visualize(df, f, ylabel, ylim=None):
    df['feature'] = f[df['id']]

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = sns.violinplot(x="bin", y="feature", ax=ax, order=BINS, data=df, 
        palette=['#FF0000', '#FFFF00', '#00CC00'], saturation=1)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Phenotype', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)

    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.xaxis.set_tick_params(length=0)

    ax.set_xlabel('', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)

    if ylim:
        ax.set_ylim(ylim)

    ax.grid(False)
    plt.setp(ax.spines.values(), linewidth=2, color='black')
    


if __name__ == "__main__":
    main()
