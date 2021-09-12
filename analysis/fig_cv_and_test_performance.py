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

import matplotlib.patches as mpatches
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
    "max_bars" : 7,

    "legend_label_size" : 42
}

ALPHA = 0.05


plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"


def main(cfg_path, test_cfg_path):
    
    
    cfg = load_cfg(cfg_path)
    test_cfg = load_cfg(test_cfg_path)

    df = create_df(cfg, test_cfg)
    model_names = [m['name'] for m in cfg['models']]
    plot_hist(df, cfg, test_cfg)
    

    plt.savefig("../tmp/test.png", bbox_inches='tight', dpi=100)


def load_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    

    if 'plot_cfg' in cfg:
        plot_cfg.update(cfg['plot_cfg'])

    if not os.path.exists(cfg['output_path']):
        os.makedirs(cfg['output_path'])

    cfg['n_models'] = len(cfg['models'])
    cfg['models'] = [m for m in cfg['models'] if m['show']]
    return cfg

def create_df(cfg, test_cfg):

    paths = [m['path'] for m in cfg['models']]
    model_names = [m['name'] for m in cfg['models']]
    colors = [m['color'] for m in cfg['models']]
    star_colors = [m['star_color'] if 'star_color' in m else m['color'] for m in cfg['models']]

    rows = []
    mid = 0
    for path, name in zip(paths, model_names):
        baccs, _, _, order = eval_funcs.collate_results(path)
        for i, bacc in enumerate(baccs):
            rows.append({
                "model" : name,
                "bacc" : bacc,
                "rep" : order[i][0],
                "fold" : order[i][1],
                "type" : "cv",
                "color" : colors[mid],
                "star_color" : star_colors[mid]
            })
        mid + 1
    
    rem_bars = plot_cfg['max_bars'] - len(model_names) 
    for i in range(rem_bars):
        rows.append({"model" : "%d" % i, "bacc" : 0, "rep" : 0, "fold" : 0, "type" : "cv", "color" : "white", "star_color" : "white" })
    
    paths = [m['path'] for m in test_cfg['models']]
    model_names = [m['name'] for m in test_cfg['models']]
    colors = [m['color'] for m in test_cfg['models']]
    star_colors = [m['star_color'] if 'star_color' in m else m['color'] for m in test_cfg['models']]

    for path, name in zip(paths, model_names):
        baccs, _, _, order = eval_funcs.collate_results(path)
        for i, bacc in enumerate(baccs):
            rows.append({
                "model" : name,
                "bacc" : bacc,
                "rep" : order[i][0],
                "fold" : order[i][1],
                "type" : "test",
                "color" : colors[mid],
                "star_color" : star_colors[mid]
            })
    
    rem_bars = plot_cfg['max_bars'] - len(model_names) 
    for i in range(rem_bars):
        rows.append({"model" : "%d" % i, "bacc" : 0, "rep" : 0, "fold" : 0, "type" : "test", "color" : "white", "star_color" : "white" })
    
    df = pd.DataFrame(rows)
    return df

def plot_hist(df, cfg, test_cfg):
    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    model_name_fsizes = [m.get('fsize', plot_cfg['bar_label_size']) for m in cfg['models']]
    star_colors = [m['star_color'] if 'star_color' in m else m['color'] for m in cfg['models']]
    
    bar = sns.barplot(x="model", 
        y="bacc", 
        hue="type",
        data=df, 
        ci="sd",
        errwidth=5,
        errcolor='black',
        saturation=1)
    ax.legend().remove()

    test_patch = mpatches.Patch(facecolor='white', fill=False, hatch='x',label='Test')
    cv_patch = mpatches.Patch(facecolor='white', fill=True, label='CV')
    
    #ax.legend(handles=[cv_patch, test_patch], fontsize=plot_cfg['legend_label_size'])

    ylim = ax.get_ylim()
    model_names = [m['name'] for m in cfg['models']]
    colors = [m['color'] for m in cfg['models']]
    model_colors = dict(zip(model_names, colors))

    sdf = df[df['type'] == 'cv']
    model_bacc = dict(zip(sdf['model'], sdf['bacc']))

    num_comparisons = cfg['n_models'] * (cfg['n_models'] - 1) / 2
    adjusted_alpha = ALPHA / num_comparisons

    for i, b in enumerate(bar.patches):
        
        model_id = i % plot_cfg['max_bars']
        type = i // plot_cfg['max_bars']
        
        if model_id >= len(model_names):
            continue 
    
        model = model_names[model_id]

        print("Model ", model, "Type ", type)

        x,y = b.get_xy()
        height = b.get_height()
        width = b.get_width()


        b.set_facecolor(model_colors[model_names[model_id]])

        if type > 0:
            b.set_hatch('x')
            b.set_linewidth(0)
            b.set_alpha(0.5)
        else:
            ax.text(model_id, model_bacc[model_names[model_id]] / 2, model_names[model_id], rotation=90, ha="center", va="center", fontsize=model_name_fsizes[i], weight='bold')

            a_ix = sdf['model'] == model_names[model_id]
            a_bacc = sdf[a_ix].sort_values(by=['rep', 'fold'])['bacc']

            offset_val = 0.05 * ylim[1]
            incr_val = 0.06 * ylim[1]

            yoffset = offset_val
            for j in range(model_id+1, len(model_names)):
                b_ix = sdf['model'] == model_names[j]
                b_bacc = sdf[b_ix].sort_values(by=['rep', 'fold'])['bacc']
                #print(model_names[i])
                #print("%d %d" % (len(a_bacc), len(b_bacc)))

                statistic, pvalue = scipy.stats.ttest_rel(a_bacc, b_bacc)

                if pvalue < adjusted_alpha:
                    stars = '*' * eval_funcs.compute_stars(pvalue, adjusted_alpha)

                    target_color = star_colors[j]
                    ax.text(i - width/2, np.mean(a_bacc) + np.std(a_bacc)/2 + yoffset, stars, color=target_color, ha="center", va="center", weight='bold', fontsize=plot_cfg['stars_label_size'])
                    yoffset += incr_val
    ax.set_xlim([-0.75, plot_cfg['max_bars'] ])
    ax.set_ylim(test_cfg['ylim'])
    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel("Balanced Accuracy", fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.yaxis.set_tick_params(length=10, width=1, which='both')

    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.savefig(output_path, bbox_inches='tight', dpi=100)

    # plt.show()

def overall_bacc(cfg, df):
    output_path = cfg['output_path']

    paths = [m['path'] for m in cfg['models']]
    model_names = [m['name'] for m in cfg['models']]
    model_name_fsizes = [m.get('fsize', plot_cfg['bar_label_size']) for m in cfg['models']]
    colors = [m['color'] for m in cfg['models']]
    star_colors = [m['star_color'] if 'star_color' in m else m['color'] for m in cfg['models']]
    
    all_baccs = []


    num_comparisons = cfg['n_models'] * (cfg['n_models'] - 1) / 2
    adjusted_alpha = ALPHA / num_comparisons

    ylim = cfg['ylim']
    
    g = sns.catplot(x="model", y="bacc", hue="type", data=df,
        kind="bar", ci="sd",
        height=10,
        aspect=cfg.get('aspect', 1),
        palette=colors,
        edgecolor='black',
        errwidth=5,
        errcolor='black',
        linewidth=plot_cfg["bar_border_size"],
        saturation=1)


    axes = g.axes

    # for ax in axes:
    #     # plot model names
    #     for i, m in enumerate(model_names):
    #         a_ix = df['model'] == m
    #         bacc = (np.mean(df[a_ix]['bacc']) - np.std(df[a_ix]['bacc'])) / 2
    #         ax.text(i, bacc, m, rotation=90, ha="center", va="center", fontsize=model_name_fsizes[i], weight='bold')


    #     # plot pvalues
    #     offset_val = 0.05 * ylim[1]
    #     incr_val = 0.04 * ylim[1]
    #     for i in range(len(model_names)):
    #         a_ix = df['model'] == model_names[i]
    #         a_bacc = df[a_ix].sort_values(by=['rep', 'fold'])['bacc']

    #         yoffset = offset_val
    #         for j in range(i+1, len(model_names)):
    #             b_ix = df['model'] == model_names[j]
    #             b_bacc = df[b_ix].sort_values(by=['rep', 'fold'])['bacc']
    #             print(model_names[i])
    #             print("%d %d" % (len(a_bacc), len(b_bacc)))

    #             statistic, pvalue = scipy.stats.ttest_rel(a_bacc, b_bacc)

    #             if pvalue < adjusted_alpha:
    #                 stars = '*' * eval_funcs.compute_stars(pvalue, adjusted_alpha)

    #                 target_color = star_colors[j]
    #                 ax.text(i, np.mean(a_bacc) + np.std(a_bacc)/2 + yoffset, stars, color=target_color, ha="center", va="center", weight='bold', fontsize=plot_cfg['stars_label_size'])
    #                 yoffset += incr_val
            
    #     ax.yaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"])
    #     ax.xaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"])
    #     ax.set_xlabel('')
    #     ax.set_ylabel('Balanced Accuracy', fontsize=plot_cfg["ylabel_size"], weight='bold')
    #     ax.set_xticklabels([])

    #     ax.yaxis.set_tick_params(length=10, width=1, which='both')
    #     ax.set_ylim(ylim)
    #     plt.setp(ax.spines.values(), linewidth=plot_cfg["border_size"], color='black')

    #plt.show()



def estimate_statistic(self, estimator, ci, n_boot, seed):

        if self.hue_names is None:
            statistic = []
            confint = []
        else:
            statistic = [[] for _ in self.plot_data]
            confint = [[] for _ in self.plot_data]

        for i, group_data in enumerate(self.plot_data):

            # Option 1: we have a single layer of grouping
            # --------------------------------------------

            if self.plot_hues is None:

                if self.plot_units is None:
                    stat_data = remove_na(group_data)
                    unit_data = None
                else:
                    unit_data = self.plot_units[i]
                    have = pd.notnull(np.c_[group_data, unit_data]).all(axis=1)
                    stat_data = group_data[have]
                    unit_data = unit_data[have]

                # Estimate a statistic from the vector of data
                if not stat_data.size:
                    statistic.append(np.nan)
                else:
                    statistic.append(estimator(stat_data))

                # Get a confidence interval for this estimate
                if ci is not None:

                    if stat_data.size < 2:
                        confint.append([np.nan, np.nan])
                        continue

                    if ci == "sd":

                        estimate = estimator(stat_data)
                        sd = np.std(stat_data) / np.sqrt(len(stat_data))
                        confint.append((estimate - sd, estimate + sd))

                    else:

                        boots = bootstrap(stat_data, func=estimator,
                                          n_boot=n_boot,
                                          units=unit_data,
                                          seed=seed)
                        confint.append(utils.ci(boots, ci))

            # Option 2: we are grouping by a hue layer
            # ----------------------------------------

            else:
                for j, hue_level in enumerate(self.hue_names):

                    if not self.plot_hues[i].size:
                        statistic[i].append(np.nan)
                        if ci is not None:
                            confint[i].append((np.nan, np.nan))
                        continue

                    hue_mask = self.plot_hues[i] == hue_level
                    if self.plot_units is None:
                        stat_data = remove_na(group_data[hue_mask])
                        unit_data = None
                    else:
                        group_units = self.plot_units[i]
                        have = pd.notnull(
                            np.c_[group_data, group_units]
                        ).all(axis=1)
                        stat_data = group_data[hue_mask & have]
                        unit_data = group_units[hue_mask & have]

                    # Estimate a statistic from the vector of data
                    if not stat_data.size:
                        statistic[i].append(np.nan)
                    else:
                        statistic[i].append(estimator(stat_data))

                    # Get a confidence interval for this estimate
                    if ci is not None:

                        if stat_data.size < 2:
                            confint[i].append([np.nan, np.nan])
                            continue

                        if ci == "sd":

                            estimate = estimator(stat_data)
                            sd = np.std(stat_data) / np.sqrt(len(stat_data))
                            confint[i].append((estimate - sd, estimate + sd))

                        else:

                            boots = bootstrap(stat_data, func=estimator,
                                              n_boot=n_boot,
                                              units=unit_data,
                                              seed=seed)
                            confint[i].append(utils.ci(boots, ci))

        # Save the resulting values for plotting
        self.statistic = np.array(statistic)
        self.confint = np.array(confint)

# catplotter = sns.categorical._CategoricalStatPlotter 
# catplotter.estimate_statistic = estimate_statistic

if __name__ == "__main__":

    cfg_path = sys.argv[1]

    main(cfg_path, sys.argv[2])

