
import seaborn as sns 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys 
import json 

BINS = np.array(['L', 'R', 'N'])

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
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'

def main(cfg):
    
    
    _, ax = plt.subplots(1, 1, figsize=(10, 10))

    for spec in cfg['species']:
        d = np.load(spec['path'])
        labels = d['feature_labels'].tolist()
        f = d['F'][:, cfg['fid']]
        f = f * d['std'][cfg['fid']] + d['mu'][cfg['fid']]
        
        if cfg['remove_zeros']:
            ix = f > 0
            f = f[ix]
        
        sns.kdeplot(f, ax=ax, linewidth=5, color=spec['color'], label=spec['name'])
        
    ax.legend(fontsize=36, frameon=False)
    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_xlabel(cfg['xlabel'], fontsize=plot_cfg['xlabel_size'], fontweight='bold')
    ax.set_ylabel('Probability', fontsize=plot_cfg['ylabel_size'], fontweight='bold')

    #ax.yaxis.set_tick_params(length=10, width=1, which='both')
    #ax.xaxis.set_tick_params(length=0)
    
    ax.set_xlim(cfg['xlim'])
    ax.grid(False)
    plt.setp(ax.spines.values(), linewidth=2, color='black')

    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)

    plt.savefig(cfg['output_path'], bbox_inches='tight', dpi=100)
    plt.close()

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    main(cfg)
