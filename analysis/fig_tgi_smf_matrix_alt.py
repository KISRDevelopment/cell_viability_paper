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
import analysis.fig_distrib_comparison_plot 

BINARY_BIN_LABELS = ['Negative', 'Neutral']
BINARY_COLORS = { 'Negative' : 'magenta', 'Neutral' : 'cyan' }
STARS_COLOR = '#007bff'

SMF_CLASSES = ['L', 'R', 'N']

def main(task_path, smf_path, output_path, ylim=[0, 70]):

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
    
    df = prepare_detailed_data(Ms)
    
    old_tick_label_size = analysis.fig_distrib_comparison_plot.plot_cfg['tick_label_size'] 
    analysis.fig_distrib_comparison_plot.plot_cfg['tick_label_size'] = 40
    analysis.fig_distrib_comparison_plot.plot_distrib(df, BINARY_COLORS, 
            '', '% Triplets', 0,
            [0, 30], STARS_COLOR, "%s_detailed.png" % output_path,
            legend_pos=(0.6, 1.05),
            stars_pos=(0.13, 0.98))

    df = prepare_summary_data(Ms)
    analysis.fig_distrib_comparison_plot.plot_cfg['tick_label_size'] = old_tick_label_size
    analysis.fig_distrib_comparison_plot.plot_distrib(df, BINARY_COLORS, 
            'Number of Normal Growth Genes', '% Triplets', 5,
            ylim, STARS_COLOR, "%s_summary" % output_path)

def prepare_detailed_data(Ms):
    rows = []
    for bin in [0, 1]:
        M = Ms[bin]
        
        vals = []
        labels = []
        for n_n in range(M.shape[0]):
            for n_r in range(4 - n_n):
                n_l = 3 - n_n - n_r
                vals.append(M[n_l, n_r, n_n])
                rows.append({
                    "x" : "L=%d\nR=%d\nN=%d" % (n_l, n_r, n_n),
                    "hue" : BINARY_BIN_LABELS[bin], 
                    "y" : M[n_l, n_r, n_n] * 100 / np.sum(M),
                    "raw_y" : M[n_l, n_r, n_n]
                })
        
    return pd.DataFrame(rows)
    
def prepare_summary_data(Ms):
    rows = []
    for bin in [0, 1]:
        M = Ms[bin]
        
        rows.append({
            "x" : "N < 2",
            "hue" : BINARY_BIN_LABELS[bin],
            "y" : np.sum(M[:, :, :2]) * 100 / np.sum(M),
            "raw_y" : np.sum(M[:, :, :2])
        })
        
        rows.append({
            "x" : "N ≥ 2",
            "hue" : BINARY_BIN_LABELS[bin],
            "y" : np.sum(M[:, :, 2:]) * 100 / np.sum(M),
            "raw_y" : np.sum(M[:,:,2:])
        })
        
    return pd.DataFrame(rows)
    
if __name__ == "__main__":
    task_path = sys.argv[1]
    smf_path = sys.argv[2]
    output_path = sys.argv[3]

    main(task_path, smf_path, output_path)
