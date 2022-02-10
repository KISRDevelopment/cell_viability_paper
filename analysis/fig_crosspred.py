import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
import sys
import json
import os 
import utils.eval_funcs as eval_funcs
import matplotlib.patches as mpatches

plot_cfg = {
    "tick_label_size" : 60,
    "xlabel_size" : 60,
    "ylabel_size" : 65,
    "border_size" : 10,
    "bar_border_size" : 2.5,
    "bar_label_size" : 55,
    "stars_label_size" : 48,
    "annot_size" : 72,
    "max_cm_classes" : 4,
    "max_bars" : 4,

    "legend_label_size" : 42
}

ALPHA = 0.05


plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"


def main(cfg_path):
    
    
    cfg = load_cfg(cfg_path)
    
    ref_df = pd.read_csv(cfg['ref_path'])
    alt_df = pd.read_csv(cfg['alt_path'])

    
    #plt.savefig("../tmp/test.png", bbox_inches='tight', dpi=100)

    df = merge_df(cfg, ref_df, alt_df)
    plot_hist(cfg, df)

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    

    if 'plot_cfg' in cfg:
        plot_cfg.update(cfg['plot_cfg'])

    return cfg

def merge_df(cfg, ref_df, alt_df):
    rows = []
    for cid, col in enumerate(cfg['cols']):

        for i in range(ref_df.shape[0]):
            rows.append({
                "val" : ref_df.iloc[i][col],
                "model" : "ref",
                "target_col" : cfg['col_labels'][cid]
            })
            rows.append({
                "val" : alt_df.iloc[i][col],
                "model" : "alt",
                "target_col" : cfg['col_labels'][cid]
            })

    return pd.DataFrame(rows)

def plot_hist(cfg, df):
    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    bar = sns.barplot(x="target_col", 
        y="val", 
        hue="model",
        data=df, 
        ci="sd",
        errwidth=5,
        errcolor='black',
        saturation=1)
    ax.legend().remove()

    alt_patch = mpatches.Patch(facecolor='white', fill=False, hatch='x',label='D-MN-Modified')
    ref_patch = mpatches.Patch(facecolor='white', fill=True, label='D-MN')
    
    #ax.legend(handles=[ref_patch, alt_patch], fontsize=plot_cfg['legend_label_size'], frameon=False, ncol=2)

    target_names = cfg['col_labels']
    target_colors = dict(zip(cfg['col_labels'], cfg['colors']))

    ix = df['model'] == 'ref'
    grouped_df = df[ix].groupby('target_col')['val'].agg('mean')

    for i, b in enumerate(bar.patches):
        
        model_id = i % plot_cfg['max_bars']
        type = i // plot_cfg['max_bars']
        
        if model_id >= len(target_names):
            continue 
    
        model = target_names[model_id]

        print("Model ", model, "Type ", type)

        width = b.get_width()


        b.set_facecolor(target_colors[target_names[model_id]])

        if type > 0:
            b.set_hatch('x')
            b.set_linewidth(0)
            b.set_alpha(0.5)
        else:
            ax.text(model_id - width/2, grouped_df.loc[target_names[model_id]] / 2, target_names[model_id], rotation=90, ha="center", va="center", 
                weight='bold', fontsize=plot_cfg['bar_label_size'])

            ref_vals = df[(df['model'] == 'ref') & (df['target_col'] == target_names[model_id])]['val']
            alt_vals = df[(df['model'] == 'alt') & (df['target_col'] == target_names[model_id])]['val']

            print(ref_vals.shape)
            print(alt_vals.shape)

            yoffset = 0.05

            statistic, pvalue = scipy.stats.ttest_rel(ref_vals, alt_vals)

            if pvalue < ALPHA:
                stars = '*' * eval_funcs.compute_stars(pvalue, ALPHA)
                ax.text(i - width/2, np.mean(ref_vals) + np.std(ref_vals)/2 + yoffset, stars, 
                color=target_colors[target_names[model_id]], ha="center", va="center", weight='bold', fontsize=plot_cfg['stars_label_size'])
                    
    ax.set_xlim([-0.75, plot_cfg['max_bars'] ])
    ax.set_ylim([0,0.85])
    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel(cfg['ylabel'], fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.yaxis.set_tick_params(length=10, width=1, which='both')

    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig("../tmp/test.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":

    cfg_path = sys.argv[1]

    main(cfg_path)

