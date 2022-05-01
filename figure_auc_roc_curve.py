import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import matplotlib 
import models.common 

plot_cfg = {
    "tick_label_size" : 60,
    "xlabel_size" : 60,
    "ylabel_size" : 65,
    "border_size" : 10,
    "legend_font_size" : 60,
    "title_size" : 60
}


plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"

def load_results(path):
    with open(path, 'r') as f:
        results = json.load(f)['results']
    
    return sorted(results, key=lambda r: r['split_id'])

def plot_auc_roc_curves(spec, klass, output_path):

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    fpr = models.common.BASE_FPR

    for model in spec['models']:
        if 'null' in model['name']:
            continue 
        
        results = load_results(model['results_path'])
        mean_tpr = np.mean([r['per_class_tpr'][klass] for r in results], axis=0)
        mean_auc_roc = np.mean([r['auc_roc'][klass] for r in results])

        ax.plot(fpr, mean_tpr, linewidth=7, color=model['color'], label="%0.2f" % mean_auc_roc)

        l = ax.legend(fontsize=plot_cfg['legend_font_size'], frameon=False)
        for p, text in enumerate(l.get_texts()):
            text.set_color(spec['models'][p]['color'])
        for line in l.get_lines():
            line.set_linewidth(0)

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
        ax.set_title(spec['classes'][klass], fontsize=plot_cfg['title_size'], fontweight='bold')
        plt.setp(ax.spines.values(), linewidth=6, color='black')

        plt.savefig(output_path, bbox_inches='tight')


if __name__ == "__main__":
    
    spec = {
        "models": [
            {
                "title": "S-Full",
                "color": "#b300ff",
                "cm_color": "#b300ff",
                "name" : "full",
                "results_path" : "../results/exp_yeast_smf/dev_test_full/results.json"
            },
            {
                "title": "S-Refined",
                "color": "#FF0000",
                "cm_color": "#FF0000",
                "name" : "refined",
                "results_path" : "../results/exp_yeast_smf/dev_test_refined/results.json"
            },
            {
                "title": "S-MN",
                "color": "#3A90FF",
                "name" : "mn",
                "results_path" : "../results/exp_yeast_smf/dev_test_mn/results.json"
            }
        ],
        "classes": [
            "Lethal",
            "Reduced growth",
            "Normal"
        ],
        "short_classes": [
            "L",
            "R",
            "N"
        ],
        "ylim": [
            0,
            0.75
        ],
        "aspect" : 1
    }

    plot_auc_roc_curves(spec, 0, '../results/exp_yeast_smf/curve_auc_roc_dev_test_L.png')
    plot_auc_roc_curves(spec, 1, '../results/exp_yeast_smf/curve_auc_roc_dev_test_R.png')
    plot_auc_roc_curves(spec, 2, '../results/exp_yeast_smf/curve_auc_roc_dev_test_N.png')
