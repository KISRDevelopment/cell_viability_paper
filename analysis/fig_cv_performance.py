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

ALPHA = 0.05


plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"


def main(cfg_path):
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    if 'plot_cfg' in cfg:
        plot_cfg.update(cfg['plot_cfg'])

    if not os.path.exists(cfg['output_path']):
        os.makedirs(cfg['output_path'])

    visible_models = [m['show'] for m in cfg['models']]
    cfg['n_models'] = len(cfg['models'])

    cfg['models'] = [m for m in cfg['models'] if m['show']]

    overall_bacc(cfg)
    per_class_figures(cfg)
    per_class_roc(cfg)
    cms(cfg)

def overall_bacc(cfg):
    output_path = cfg['output_path']

    paths = [m['path'] for m in cfg['models']]
    model_names = [m['name'] for m in cfg['models']]
    colors = [m['color'] for m in cfg['models']]
    star_colors = [m['star_color'] if 'star_color' in m else m['color'] for m in cfg['models']]
    
    all_baccs = []


    rows = []
    for path, name in zip(paths, model_names):
        baccs, _, _, order = eval_funcs.collate_results(path)
        for i, bacc in enumerate(baccs):
            rows.append({
                "model" : name,
                "bacc" : bacc,
                "rep" : order[i][0],
                "fold" : order[i][1]
            })

    df = pd.DataFrame(rows)

    num_comparisons = cfg['n_models'] * (cfg['n_models'] - 1) / 2
    adjusted_alpha = ALPHA / num_comparisons

    ylim = cfg['ylim']
    
    g = sns.catplot(x="model", y="bacc", data=df,
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

    # plot model names
    for i, m in enumerate(model_names):
        a_ix = df['model'] == m
        bacc = (np.mean(df[a_ix]['bacc']) - np.std(df[a_ix]['bacc'])) / 2
        ax.text(i, bacc, m, rotation=90, ha="center", va="center", fontsize=plot_cfg['bar_label_size'], weight='bold')


    # plot pvalues
    offset_val = 0.05 * ylim[1]
    incr_val = 0.04 * ylim[1]
    for i in range(len(model_names)):
        a_ix = df['model'] == model_names[i]
        a_bacc = df[a_ix].sort_values(by=['rep', 'fold'])['bacc']

        yoffset = offset_val
        for j in range(i+1, len(model_names)):
            b_ix = df['model'] == model_names[j]
            b_bacc = df[b_ix].sort_values(by=['rep', 'fold'])['bacc']
            print(model_names[i])
            print("%d %d" % (len(a_bacc), len(b_bacc)))

            statistic, pvalue = scipy.stats.ttest_rel(a_bacc, b_bacc)

            if pvalue < adjusted_alpha:
                stars = '*' * eval_funcs.compute_stars(pvalue, adjusted_alpha)

                target_color = star_colors[j]
                ax.text(i, np.mean(a_bacc) + np.std(a_bacc)/2 + yoffset, stars, color=target_color, ha="center", va="center", weight='bold', fontsize=plot_cfg['stars_label_size'])
                yoffset += incr_val
        
    ax.yaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"])
    ax.xaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"])
    ax.set_xlabel('')
    ax.set_ylabel('Balanced Accuracy', fontsize=plot_cfg["ylabel_size"], weight='bold')
    ax.set_xticklabels([])

    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.set_ylim(ylim)
    plt.setp(ax.spines.values(), linewidth=plot_cfg["border_size"], color='black')

    plt.savefig("%s/%s" % (cfg['output_path'], "/overall_bacc.png"), bbox_inches='tight', dpi=100)

    #plt.show()


def per_class_figures(cfg):
    output_path = cfg['output_path']
    paths = [m['path'] for m in cfg['models']]
    model_names = [m['name'] for m in cfg['models']]
    classes = cfg['classes']

    rows = []
    for path, name in zip(paths, model_names):
        r = eval_funcs.collate_results(path)
        for _, class_bacc, class_roc, order in zip(*r):
            for c in range(len(classes)):
                rows.append({
                    "model" : name,
                    "bin" : classes[c],
                    "bacc" : class_bacc[c],
                    "roc" : class_roc[c],
                    "rep" : order[0],
                    "fold" : order[1]
                })

    df = pd.DataFrame(rows)

    per_class(df, cfg, "bacc", "Balanced Accuracy", "%s/per_class_bacc" % output_path)
    per_class(df, cfg, "roc", "AUC-ROC", "%s/per_class_roc" % output_path)

def cms(cfg):
    output_path = cfg['output_path']
    paths = [m['path'] for m in cfg['models']]
    model_names = [m['name'] for m in cfg['models']]
    colors = [m['cm_color'] if 'cm_color' in m else m['color'] for m in cfg['models']]
    classes = cfg['classes']
    sclasses = cfg['short_classes']
    
    import matplotlib

    for path, name, color in zip(paths, model_names, colors):
        cm = eval_funcs.average_cm(path)

        print(path)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
            ["white",color])

        f, ax = plt.subplots(figsize=(10, 10))

        #print(cm)
        ax.imshow(cm, cmap=cmap)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, "%0.2f" % cm[i, j], ha="center", va="center", 
                fontsize=plot_cfg['annot_size'])

        # ax.set_xticks(np.arange(len(classes)))
        # ax.set_yticks(np.arange(len(classes)))

        xlabels = ax.get_xticks()
        ix = np.isin(xlabels, np.arange(len(classes)))
        xlabels = xlabels.astype(str)
        xlabels[ix] = sclasses
        xlabels[~ix] = ''

        ax.set_xticklabels(xlabels, fontsize=plot_cfg['annot_size'])
        ax.set_yticklabels(xlabels, fontsize=plot_cfg['annot_size'])
        
        ax.xaxis.set_tick_params(length=0, width=0, which='both', colors=color)
        ax.yaxis.set_tick_params(length=0, width=0, which='both', colors=color)
        ax.xaxis.tick_top()
        plt.setp(ax.spines.values(), linewidth=0)

        plt.savefig("%s/cm_%s.png" % (output_path, name), bbox_inches='tight')
        #
        # plt.show()
        
def per_class_roc(cfg):

    models = [m for m in cfg['models'] if ('show_roc' not in m) or m['show_roc']]
    output_path = cfg['output_path']
    paths = [m['path'] for m in models]
    model_names = [m['name'] for m in models]
    classes = cfg['classes']
    colors = [m['cm_color'] if 'cm_color' in m else m['color'] for m in models]


    n_classes = len(classes)
    for klass in range(n_classes-1, -1, -1):
        f, ax = plt.subplots(1, 1, figsize=(10, 10))

        for p, path in enumerate(paths):
            fpr, roc_curve = eval_funcs.average_roc_curve(path, klass)
            ax.plot(fpr, roc_curve, linewidth=5, color=colors[p])

        ax.plot(fpr, fpr, linewidth=2, color='black', linestyle='dashed')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
        ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
        ax.set_xlabel('False Positive Rate', fontsize=plot_cfg['xlabel_size'], weight='heavy')
        ax.set_ylabel('True Positive Rate', fontsize=plot_cfg['ylabel_size'], weight='heavy')
        ax.yaxis.set_tick_params(length=10, width=1, which='both')
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)

        plt.setp(ax.spines.values(), linewidth=6, color='black')

        plt.savefig("%s/roc_curve_%s.png" % (output_path, classes[klass]), bbox_inches='tight', dpi=100)


def per_class(df, cfg, y, ylabel, output_path):
    ALPHA = 0.05
    model_names = [m['name'] for m in cfg['models']]
    colors = [m['color'] for m in cfg['models']]
    classes = cfg['classes']
    star_colors = [m['star_color'] if 'star_color' in m else m['color'] for m in cfg['models']]

    all_vals = []
    for m in model_names:
        vals = [np.mean(df[(df['bin'] == c) & (df['model'] == m)][y]) for c in classes]
        all_vals.extend(vals)
    
    max_val = np.max(all_vals)
    
    print("%s max val: %f" % (output_path, max_val*1.25))
    ylim = [0, max_val*1.25]
    
    offset_val = 0.05 * ylim[1]
    incr_val = 0.04 * ylim[1]
    print("Offset: %0.2f, incr: %0.2f" % (offset_val, incr_val))
    for i, c in enumerate(classes):
        sdf = df[df['bin'] == c]

        #f, ax = plt.subplots(1, 1, figsize=(10,10))

        g = sns.catplot(x="model", y=y, data=sdf,
            kind="bar", ci="sd",
            estimator=np.mean,
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
            a_ix = sdf['model'] == m
            val = (np.mean(sdf[a_ix][y]) - np.std(sdf[a_ix][y])) / 2
            ax.text(i, val, m, rotation=90, ha="center", va="center", fontsize=plot_cfg['bar_label_size'], weight='bold')


        num_comparisons = cfg['n_models'] * (cfg['n_models']  - 1) / 2
        adjusted_alpha = ALPHA / num_comparisons

        for i in range(len(model_names)):
            a_ix = sdf['model'] == model_names[i]
            a_bacc = sdf[a_ix].sort_values(by=['rep', 'fold'])[y]
            print("Model %s %s = %0.2f, %0.2f" % (model_names[i], c, np.mean(a_bacc), np.median(a_bacc)))
            yoffset = offset_val
            for j in range(i+1, len(model_names)):
                b_ix = sdf['model'] == model_names[j]
                b_bacc = sdf[b_ix].sort_values(by=['rep', 'fold'])[y]
                statistic, pvalue = scipy.stats.ttest_rel(a_bacc, b_bacc)
                if pvalue < adjusted_alpha:
                    stars = '*' * eval_funcs.compute_stars(pvalue, adjusted_alpha)
                    target_color = star_colors[j]
                    ax.text(i, np.mean(a_bacc) + np.std(a_bacc) + yoffset, stars, 
                        color=target_color, ha="center", va="top", weight='bold', fontsize=plot_cfg['stars_label_size'])
                    yoffset += incr_val

        ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
        ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
        ax.set_ylabel(ylabel, fontsize=plot_cfg['ylabel_size'], fontweight='bold')
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.yaxis.set_tick_params(length=10, width=1, which='both')
        ax.set_ylim(ylim)

        plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.savefig("%s_%s.png" % (output_path, c), bbox_inches='tight', dpi=100)




if __name__ == "__main__":

    cfg_path = sys.argv[1]

    main(cfg_path)

