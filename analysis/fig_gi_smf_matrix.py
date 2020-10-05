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

BINARY_BIN_LABELS = ['Interacting', 'Neutral']
BINARY_COLORS = ['#FF00DA', 'orange']

BIN_LABELS = ['Negative', 'Neutral', 'Positive', 'Suppression']
COLORS = ['#FF0000', 'orange', '#00CC00', '#3d77ff']

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


def main(task_path, smf_path, output_path, binary=False):

    df = pd.read_csv(task_path)
    
    smf_df = pd.read_csv(smf_path)

    smf = dict(zip(smf_df['id'], smf_df['bin']))
    
    genes_with_smf = set(smf_df['id'])

    with_smf_ix = (df['a_id'].isin(genes_with_smf)) & (df['b_id'].isin(genes_with_smf))
    print("# pairs without smf: %d" % np.sum(~with_smf_ix))

    labels = BIN_LABELS
    colors = COLORS
    if binary:
        df['bin'] = (df['bin'] == 1).astype(int)
        labels = BINARY_BIN_LABELS
        colors = BINARY_COLORS

    Ms = []
    for bin in range(len(labels)):
        ix = (df['bin'] == bin) & with_smf_ix
        print("# observation in %d: %d" % (bin, np.sum(ix)))

        sdf = df[ix]

        a_smf_bin = np.array([smf[i] for i in sdf['a_id']]).astype(int)
        b_smf_bin = np.array([smf[i] for i in sdf['b_id']]).astype(int)
        
        M = np.zeros((3, 3))
        for i in range(len(a_smf_bin)):
            abin = a_smf_bin[i]
            bbin = b_smf_bin[i]

            M[abin, bbin] += 1
        
        M = np.triu(M) + np.tril(M, k=-1).T
        Ms.append(M)

        Mnormed = M / np.sum(M)


        f, ax = plt.subplots(figsize=(10, 10))

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["white",colors[bin]])

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
        
        ax.xaxis.set_tick_params(length=0, width=0, which='both', colors=colors[bin])
        ax.yaxis.set_tick_params(length=0, width=0, which='both', colors=colors[bin])
        ax.xaxis.tick_top()
        plt.setp(ax.spines.values(), linewidth=0)

        plt.savefig("%s_%s.png" % (output_path, labels[bin]), bbox_inches='tight', dpi=100)


        plt.close()
        #plt.show()

    M = np.array(Ms)
    Mtot = np.sum(M, axis=0)

    for i in range(len(Ms)):
        print(M[i, :, :] / Mtot)
        
if __name__ == "__main__":
    task_path = sys.argv[1]
    smf_path = sys.argv[2]
    output_path = sys.argv[3]

    main(task_path, smf_path, output_path, True)
