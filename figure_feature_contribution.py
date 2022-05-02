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
    "max_bars" : 7,
    "legend_label_size" : 42
}

ALPHA = 0.05

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"

def generate_figures(spec, path, output_path):

    df = create_plot_df(spec, path) 
    plot_df(spec, df, output_path)

def create_plot_df(spec, path):

    results_path = os.path.join(path, spec['models'][0]['name'], "results.json")
        
    cv_results = load_results(results_path)
    
    # reference baccs
    baccs = [r['bacc'] for r in cv_results]

    rows = []
    for model in spec['models'][1:]:
        results_path = os.path.join(path, model['name'], "results.json")
        
        cv_results = load_results(results_path)
        
        for i, r in enumerate(cv_results):
            rows.append({
                "model" : model['title'],
                "bacc" : r['bacc'],
                "drop_perc" : (baccs[i] - r['bacc']) * 100 / baccs[i],
                "color" : model['color'],
                "star_color" : model.get('star_color', model['color'])
            })
    
    df = pd.DataFrame(rows)
    print(df)
    return df 

def load_results(path):
    with open(path, 'r') as f:
        results = json.load(f)['results']
    
    return sorted(results, key=lambda r: r['split_id'])

def plot_df(spec, df, output_path):
    n_models = len(spec['models']) - 1
    models = spec['models'][1:]

    num_comparisons = n_models * (n_models - 1) / 2
    adjusted_alpha = ALPHA / num_comparisons

    ylim = spec['ylim']
    
    g = sns.catplot(x="model", 
        y="drop_perc", 
        data=df,
        kind="bar", ci="sd",
        height=10,
        aspect=spec.get('aspect', 1),
        palette=[m['color'] for m in models],
        edgecolor='black',
        errwidth=5,
        errcolor='black',
        linewidth=plot_cfg["bar_border_size"],
        saturation=1)

    ax = g.ax

    # plot pvalues
    offset_val = 0.01 * ylim[1]
    incr_val = 0.04 * ylim[1]

    for i, model in enumerate(models):
        a_ix = df['model'] == model['title']
        a_bacc = df[a_ix]['bacc']
        a_drop = df[a_ix]['drop_perc']

        yoffset = offset_val

        for j in range(i+1, n_models):
            b_ix = df['model'] == models[j]['title']
            b_bacc = df[b_ix]['bacc']
            
            statistic, pvalue = scipy.stats.ttest_rel(a_bacc, b_bacc)

            if pvalue < adjusted_alpha:
                stars = '*' * utils.stats.compute_stars(pvalue, adjusted_alpha)

                target_color = models[j].get('star_color', models[j]['color'])

                ax.text(i,np.mean(a_drop) + np.std(a_drop) + yoffset, stars, color=target_color, ha="center", va="center", weight='bold', fontsize=plot_cfg['stars_label_size'])
                yoffset += incr_val
        
    ax.yaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"])
    ax.xaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"], rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('% Drop', fontsize=plot_cfg["ylabel_size"], weight='bold')
    

    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.set_ylim(ylim)
    plt.setp(ax.spines.values(), linewidth=plot_cfg["border_size"], color='black')

    plt.savefig(output_path, bbox_inches='tight')

if __name__ == "__main__":

    spec = {
        "models": [
            {
                "title": "S-MN",
                "color": "#FF0000",
                "name" : "smf"
            },
            {
                "title": "No LID",
                "color": "magenta",
                "name" : "smf_no_lid"
            },
            {
                "title": "No sGO",
                "color": "cyan",
                "name" : "smf_no_sgo"
            },
            {
                "title": "No Homology",
                "color": "#d62728",
                "name" : "smf_no_redundancy"
            },
        ],
        "ylim" : [0,30],
        "aspect" : 1
    }

    generate_figures(spec, "../results/exp_mn_feature_contribution", 
        "../results/exp_mn_feature_contribution/bacc_drop.png")
