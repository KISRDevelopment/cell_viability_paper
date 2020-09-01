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

BIN_LABELS = ['Interacting', 'Neutral']
COLORS = ['#FF00DA', 'orange']

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"

SMF_CLASSES = ['L', 'R', 'N']

plot_cfg = {
    "tick_label_size" : 50,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 82
}


def main(task_path, smf_path, output_path):

    df = pd.read_csv(task_path)
    A = np.array(df[['a_id', 'b_id']]).T
    np.random.shuffle(A)
    df[['a_id', 'b_id']] = A.T 

    smf_df = pd.read_csv(smf_path)

    smf = dict(zip(smf_df['id'], smf_df['bin']))
    
    genes_with_smf = set(smf_df['id'])

    with_smf_ix = (df['a_id'].isin(genes_with_smf)) & (df['b_id'].isin(genes_with_smf))
    print("# pairs without smf: %d" % np.sum(~with_smf_ix))

    for bin in range(len(BIN_LABELS)):
        if bin == 0:
            ix = (df['bin'] != 1) & with_smf_ix
            print("# observation in %d: %d, # obs with smf: %d" % (bin, np.sum(df['bin'] != 1), np.sum(ix)))

        else:
            ix = (df['bin'] == 1) & with_smf_ix
            print("# observation in %d: %d, # obs with smf: %d" % (bin, np.sum(df['bin'] == bin), np.sum(ix)))

        sdf = df[ix]

        a_smf_bin = np.array([smf[i] for i in sdf['a_id']]).astype(int)
        b_smf_bin = np.array([smf[i] for i in sdf['b_id']]).astype(int)
        
        M = np.zeros((3, 3))
        for i in range(len(a_smf_bin)):
            abin = a_smf_bin[i]
            bbin = b_smf_bin[i]

            M[abin, bbin] += 1

                
        M = np.triu(M)
        Mnormed = M / np.sum(M)


        f, ax = plt.subplots(figsize=(10, 10))

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["white",COLORS[bin]])

        ax.imshow(Mnormed, cmap=cmap)
        for i in range(M.shape[0]):
            for j in range(i,M.shape[0]):
                ax.text(j, i, "%0.2f" % Mnormed[i, j], ha="center", va="center", 
                fontsize=plot_cfg['annot_size'])

        xlabels = ax.get_xticks()
        ix = np.isin(xlabels, np.arange(Mnormed.shape[0]))
        xlabels = xlabels.astype(str)
        xlabels[ix] = SMF_CLASSES
        xlabels[~ix] = ''

        ax.set_xticklabels(xlabels, fontsize=plot_cfg['annot_size'])
        ax.set_yticklabels(xlabels, fontsize=plot_cfg['annot_size'])
        
        ax.xaxis.set_tick_params(length=0, width=0, which='both', colors=COLORS[bin])
        ax.yaxis.set_tick_params(length=0, width=0, which='both', colors=COLORS[bin])
        ax.xaxis.tick_top()
        plt.setp(ax.spines.values(), linewidth=0)

        plt.savefig("%s/smf_interacting_matrix_%s.png" % (output_path, BIN_LABELS[bin]), bbox_inches='tight', dpi=100)


        plt.show()

if __name__ == "__main__":
    task_path = sys.argv[1]
    smf_path = sys.argv[2]
    output_path = sys.argv[3]

    main(task_path, smf_path, output_path)