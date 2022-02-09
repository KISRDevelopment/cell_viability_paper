#
# Figures:
#   1. Overall BACC [ok]
#   2. Per class BACC and ROC [ok]
#   3. Per class ROC curves [ok]
#   4. Confusion matrices [ok]
#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import scipy.stats
import seaborn as sns
import sys
import json
import os 
import utils.eval_funcs as eval_funcs
from seaborn.utils import remove_na

plot_cfg = {
    "tick_label_size" : 60,
    "xlabel_size" : 60,
    "ylabel_size" : 65,
    "border_size" : 10,
    "bar_border_size" : 2.5,
    "bar_label_size" : 65,
    "stars_label_size" : 48,
    "annot_size" : 72,
    "max_cm_classes" : 4,
    "max_bars" : 5,
    "legend_font_size" : 60
}

ALPHA = 0.05


plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"


def main(cfg_path):
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    if 'plot_cfg' in cfg:
        plot_cfg.update(cfg['plot_cfg'])

    cfg['n_models'] = len(cfg['models'])

    overall_bacc(cfg)
    
def overall_bacc(cfg):
    output_path = cfg['output_path']

    paths = [m['path'] for m in cfg['models']]
    model_names = [m['name'] for m in cfg['models']]
    model_name_fsizes = [m.get('fsize', plot_cfg['bar_label_size']) for m in cfg['models']]
    colors = [m['color'] for m in cfg['models']]
    star_colors = [m['star_color'] if 'star_color' in m else m['color'] for m in cfg['models']]
    
    all_baccs = []


    ref_baccs, _, _, order = eval_funcs.collate_results(cfg['ref_model']['path'])
    ref_rows = []
    for i, bacc in enumerate(ref_baccs):
            ref_rows.append({
                "bacc" : bacc,
                "rep" : order[i][0],
                "fold" : order[i][1],
                "repfold" : order[i]
            })
    ref_df = pd.DataFrame(ref_rows)
    ref_df.index = pd.MultiIndex.from_tuples(ref_df['repfold'])

    rows = []
    
    for path, name in zip(paths, model_names):
        baccs, _, _, order = eval_funcs.collate_results(path)
        for i, bacc in enumerate(baccs):
            rows.append({
                "model" : name,
                "drop_perc" : (ref_df.loc[tuple(order[i]),'bacc']- bacc) * 100 / ref_df.loc[tuple(order[i]),'bacc'],
                "bacc" : bacc,
                "rep" : order[i][0],
                "fold" : order[i][1]
            })
    
    rem_bars = plot_cfg['max_bars'] - len(model_names) 
    for i in range(rem_bars):
        rows.append({"model" : "%d" % i, "bacc" : 0, "drop_perc" : 0, "rep" : 0, "fold" : 0 })
    
    df = pd.DataFrame(rows)
    print(df)
    num_comparisons = cfg['n_models'] * (cfg['n_models'] - 1) / 2
    adjusted_alpha = ALPHA / num_comparisons

    ylim = cfg['ylim']
    
    g = sns.catplot(x="model", y="drop_perc", data=df,
        kind="bar", ci="sd",
        height=10,
        aspect=cfg.get('aspect', 1),
        palette=colors,
        edgecolor='black',
        errwidth=5,
        errcolor='black',
        linewidth=plot_cfg["bar_border_size"],
        saturation=1)

    ax = g.ax

    for i, m in enumerate(model_names):
        a_ix = df['model'] == m
        bacc = (np.mean(df[a_ix]['drop_perc']) - np.std(df[a_ix]['drop_perc'])) / 2
        ax.text(i, -0.5, m, rotation=90, ha="center", va="top", fontsize=model_name_fsizes[i], weight='bold')


    # plot pvalues
    offset_val = 0.01 * ylim[1]
    incr_val = 0.04 * ylim[1]

    adjusted_alpha_to_ref = ALPHA / cfg['n_models']

    for i in range(len(model_names)):
        a_ix = df['model'] == model_names[i]
        a_bacc = df[a_ix].sort_values(by=['rep', 'fold'])['bacc']
        a_drop = df[a_ix].sort_values(by=['rep', 'fold'])['drop_perc']
        
        yoffset = offset_val
        b_bacc = ref_df.sort_values(by=['rep', 'fold'])['bacc']
        statistic, pvalue = scipy.stats.ttest_rel(a_bacc, b_bacc)
        
        if pvalue < adjusted_alpha_to_ref:
            stars = '*' * eval_funcs.compute_stars(pvalue, adjusted_alpha)
            
            target_color = cfg['ref_model']['color']
            ax.text(i, np.mean(a_drop) + np.std(a_drop) + yoffset, stars, color=target_color, ha="center", va="center", weight='bold', fontsize=plot_cfg['stars_label_size'])
            yoffset += incr_val
        

        for j in range(i+1, len(model_names)):
            b_ix = df['model'] == model_names[j]
            b_bacc = df[b_ix].sort_values(by=['rep', 'fold'])['bacc']
            print(model_names[i])
            print("%d %d" % (len(a_bacc), len(b_bacc)))

            statistic, pvalue = scipy.stats.ttest_rel(a_bacc, b_bacc)

            if pvalue < adjusted_alpha:
                stars = '*' * eval_funcs.compute_stars(pvalue, adjusted_alpha)

                target_color = star_colors[j]
                ax.text(i,np.mean(a_drop) + np.std(a_drop) + yoffset, stars, color=target_color, ha="center", va="center", weight='bold', fontsize=plot_cfg['stars_label_size'])
                yoffset += incr_val
        
    ax.yaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"])
    ax.xaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"])
    ax.set_xlabel('')
    ax.set_ylabel('% Drop', fontsize=plot_cfg["ylabel_size"], weight='bold')
    ax.set_xticklabels([])

    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.set_ylim(ylim)
    plt.setp(ax.spines.values(), linewidth=plot_cfg["border_size"], color='black')

    plt.savefig(cfg['output_path'], bbox_inches='tight', dpi=100)


if __name__ == "__main__":

    cfg_path = sys.argv[1]

    main(cfg_path)

