import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import utils.stats
import seaborn as sns
import json
import os 

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
    "legend_label_size" : 42
}

ALPHA = 0.05

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"

def generate_figures(spec, path, output_path):

    df = create_plot_df(spec, path) 
    
    plot_df(spec, df, output_path)

def create_plot_df(spec, path):

    rows = []
    for model in spec['models']:
        
        for i, results_path in enumerate(model['results_paths']):
            cv_results = load_results(results_path)
            
            for r in cv_results:
                rows.append({
                    "model" : model['title'],
                    "bacc" : r['bacc'],
                    "type" : i,
                    "color" : model['color'],
                    "star_color" : model.get('star_color', model['color'])
                })
    
    df = pd.DataFrame(rows)

    return df 

def load_results(path):
    with open(path, 'r') as f:
        results = json.load(f)['results']
    
    return sorted(results, key=lambda r: r['split_id'])

def plot_df(spec, df, output_path):
    f, ax = plt.subplots(1, 1, figsize=(20, 10))
    n_models = len(spec['models'])

    adjusted_alpha = ALPHA 
    bar = sns.barplot(x="model", 
        y="bacc",
        hue="type",
        data=df, 
        ci="sd",
        errwidth=5,
        errcolor='black',
        saturation=1)
    ax.legend().remove()
    ylim = ax.get_ylim()
    
    for i, b in enumerate(bar.patches):
        
        model_id = i % n_models
        type = i // n_models
        
        model = spec['models'][model_id]

        width = b.get_width()

        b.set_facecolor(model['color'])

        if type > 0:
            b.set_hatch('x')
            b.set_linewidth(0)
            b.set_alpha(0.5)
        else:

            bacc = b.get_height()
            
            ax.text(model_id - width/2, bacc / 2, model['title'], rotation=90, ha="center", va="center", fontsize=model.get('fsize', plot_cfg['bar_label_size']), weight='bold')

            a_ix = (df['model'] == model['title']) & (df['type'] == 0)
            a_bacc = df[a_ix]['bacc']

            offset_val = 0.05 * ylim[1]
            incr_val = 0.06 * ylim[1]

            yoffset = offset_val
            
            b_ix = (df['model'] == spec['models'][model_id]['title']) & (df['type'] == 1)
            b_bacc = df[b_ix]['bacc']
            statistic, pvalue = scipy.stats.ttest_rel(a_bacc, b_bacc)
                
            if pvalue < adjusted_alpha:
                stars = '*' * utils.stats.compute_stars(pvalue, adjusted_alpha)
                target_color = spec['models'][model_id].get('star_color', spec['models'][model_id]['color'])
                ax.text(i - width/2, np.mean(a_bacc) + np.std(a_bacc)/2 + yoffset, 
                            stars, color=target_color, 
                            ha="center", 
                            va="center", 
                            weight='bold', fontsize=plot_cfg['stars_label_size'])
                yoffset += incr_val
    
    ax.set_xlim([-0.75, plot_cfg['max_bars'] ])
    ax.set_ylim(spec['ylim'])
    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel("Balanced Accuracy", fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.yaxis.set_tick_params(length=10, width=1, which='both')

    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(output_path, bbox_inches='tight')

if __name__ == "__main__":

    spec = {
        "models": [
            {
                "title": "GI",
                "color": "#1f77b4",
                "cm_color": "#1f77b4",
                "results_paths" : [
                    "../results/exp_cross_prediction/mn/results.json",
                    "../results/exp_cross_prediction/mn_strict/results.json",
                ],
                "name" : "gi"
            },
            {
                "title": "Coprecipitation",
                "color": "#ff7f0e",
                "cm_color": "#ff7f0e",
                "results_paths" : [
                    "../results/exp_cross_prediction/mn_rel_not_ppc.json",
                    "../results/exp_cross_prediction/mn_strict_rel_not_ppc.json"
                ],
                "fsize" : 50,
                "name" : "ppc"
            },
            {
                "title": "Phopshorylation",
                "color": "#2ca02c",
                "results_paths" : [
                    "../results/exp_cross_prediction/mn_rel_not_phospho.json",
                    "../results/exp_cross_prediction/mn_strict_rel_not_phospho.json"
                ],
                "name" : "phospho"
            },
            {
                "title": "Transcription",
                "color": "#d62728",
                "results_paths" : [
                    "../results/exp_cross_prediction/mn_rel_not_trans.json",
                    "../results/exp_cross_prediction/mn_strict_rel_not_trans.json",
                ],
                "fsize" : 50,
                "name" : "trans"
            }
        ],
        "classes": [
            "Interacting",
            "Neutral"
        ],
        "short_classes": [
            "I",
            "N"
        ],
        "ylim" : [0,0.8],
        "aspect" : 1
    }


    generate_figures(spec, "../results/exp_cross_prediction", "../results/exp_cross_prediction/figures/overall_bacc.png")
