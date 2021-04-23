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

BINARY_BIN_LABELS = ['Negative', 'Neutral']
BINARY_COLORS = ['magenta', 'cyan']

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

    labels = BINARY_BIN_LABELS
    colors = BINARY_COLORS

    Ms = []
    for bin in range(len(labels)):
        ix = (df['bin'] == bin) & with_smf_ix
        print("# observation in %d: %d" % (bin, np.sum(ix)))

        sdf = df[ix]

        a_smf_bin = np.array([smf[i] for i in sdf['a_id']]).astype(int)
        b_smf_bin = np.array([smf[i] for i in sdf['b_id']]).astype(int)
        c_smf_bin = np.array([smf[i] for i in sdf['c_id']]).astype(int)

        M = np.zeros((len(SMF_CLASSES) * (len(SMF_CLASSES) - 1), len(SMF_CLASSES)))
        for i in range(len(a_smf_bin)):
            abin = a_smf_bin[i]
            bbin = b_smf_bin[i]
            cbin = c_smf_bin[i]


            M[MAPPING[(abin, bbin)], cbin] += 1
        
        Ms.append(M)

        Mnormed = M / np.sum(M)
        Mnormed = Mnormed[ROW_ORDER, :]

        f, ax = plt.subplots(figsize=(20, 20))

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["white",colors[bin]])

        ax.imshow(Mnormed, cmap=cmap)
        for i in range(M.shape[0]):
            for j in range(0, M.shape[1]):
                ax.text(j, i, "%0.2f" % Mnormed[i, j], ha="center", va="center", 
                fontsize=plot_cfg['annot_size'])

        ax.set_yticks(range(0, len(ROW_LABELS)))
        ax.set_yticklabels(np.array(ROW_LABELS)[ROW_ORDER], fontsize=plot_cfg['annot_size'])
        
        ax.set_xticks(range(0, Mnormed.shape[1]))
        ax.set_xticklabels(SMF_CLASSES, fontsize=plot_cfg['annot_size'])

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
