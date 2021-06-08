import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches
plot_cfg = {
    "tick_label_size" : 50,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 32,
    "stars_label_size" : 48,
    "annot_size" : 72,
    "max_cm_classes" : 4,
    "legend_size" : 42,
    "max_bars" : 4
}

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
SMF_RESULTS = [
    {
        "name" : "OR Model",
        "color" : "#00CC00",
        "auc-roc" : [0.85, 0.61, 0.79],
    },
    {
        "name" : "Campos et al. (2019)",
        "color" : "#b49fdc",
        "auc-roc" : [0.75, np.nan, 0.75],
        "hatch" : "",
        "bar_labels" : ["", "", "Not Well Defined"]
    },
    {
        "name" : "Lei et al. (2018)",
        "color" : "#c5ebfe",
        "auc-roc" : [0.74, np.nan, 0.74],
        "hatch" : ".",
        "bar_labels" : ["", "", "Not Well Defined"]
    },
    {
        "name" : "Mistry et al. (2017)",
        "color" : "#fefd97",
        "auc-roc" : [0.68, np.nan, 0.68],
        "hatch" : ".",
        "bar_labels" : ["", "", "Not Well Defined"]
    },
    {
        "name" : "Luo & Qi (2015)",
        "color" : "#fec9a7",
        "auc-roc" : [0.66, np.nan, 0.66],
        "hatch" : ".",
        "bar_labels" : ["", "", "Not Well Defined"]
    },
    {
        "name" : "Del Rio et al. (2009)",
        "color" : "#f197c0",
        "auc-roc" : [0.67, np.nan, 0.67],
        "hatch" : ".",
        "bar_labels" : ["", "", "Not Well Defined"]
    }
]
SMF_BINS = ['Lethal/Essential', 'Reduced Growth', 'Normal']

GI_RESULTS = [
    {
        "name" : "MN Model",
        "color" : "#3A90FF",
        "auc-roc" : [0.79, 0.86, 0.88, 0.86],
    },
   
    {
        "name" : "Benstead-Hume et al. (2019)",
        "color" : "#57c7ff",
        "auc-roc" : [0.88, 0.88, np.nan, np.nan],
        "hatch" : "x",
        "bar_labels" : ["", "Not Well Defined", "", ""]
    },
    {
        "name" : "Wu et al. (2014)",
        "color" : "#fec9a7",
        "auc-roc" : [0.86, 0.86, np.nan, np.nan],
        "hatch" : "x",
        "bar_labels" : ["", "Not Well Defined", "", ""]
    },
    {
        "name" : "Wong et al. (2004)",
        "color" : "#f197c0",
        "auc-roc" : [0.80, 0.80, np.nan, np.nan],
        "bar_labels" : ["", "Not Well Defined", "", ""]
    },
]
GI_BINS = ['Negative', 'Neutral', 'Positive', 'Suppression']


BINARY_SMF_RESULTS = [
    {
        "name" : "OR Model",
        "color" : "#00CC00",
        "auc-roc" : [0.89, np.nan, np.nan],
    },
    {
        "name" : "Campos et al. (2019)",
        "color" : "#b49fdc",
        "auc-roc" : [0.75, np.nan, np.nan],
    },
    {
        "name" : "Lei et al. (2018)",
        "color" : "#c5ebfe",
        "auc-roc" : [0.74, np.nan, np.nan],
        "hatch" : "."
    },
    {
        "name" : "Mistry et al. (2017)",
        "color" : "#fefd97",
        "auc-roc" : [0.68, np.nan, np.nan],
        "hatch" : "."
    },
    {
        "name" : "Luo & Qi (2015)",
        "color" : "#fec9a7",
        "auc-roc" : [0.66, np.nan, np.nan],
        "hatch" : "."
    },
    {
        "name" : "Del Rio et al. (2009)",
        "color" : "#f197c0",
        "auc-roc" : [0.67, np.nan, np.nan],
        "hatch" : "."
    }
]
BINARY_SMF_BINS = ['Lethal (Essential)', ' ', '  ']


BINARY_GI_RESULTS = [
    {
        "name" : "MN Model",
        "color" : "#3A90FF",
        "auc-roc" : [0.86, np.nan, np.nan, np.nan],
    },
    {
        "name" : "Benstead-Hume et al. (2019)",
        "color" : "#57c7ff",
        "auc-roc" : [0.88, np.nan, np.nan, np.nan],
        "hatch" :  "x",
    },
    {
        "name" : "Wu et al. (2014)",
        "color" : "#fec9a7",
        "auc-roc" : [0.86, np.nan, np.nan, np.nan],
        "hatch" : "x"
    },
    {
        "name" : "Wong et al. (2004)",
        "color" : "#f197c0",
        "auc-roc" : [0.80, np.nan, np.nan, np.nan],
        "hatch" : ""
    },
]
BINARY_GI_BINS = ['Interacting', ' ', '  ', '    ']

def main():

    visualize_results(SMF_RESULTS, SMF_BINS, "../tmp/smf_related_work.png")
    visualize_results(GI_RESULTS, GI_BINS, "../tmp/gi_related_work.png")
    visualize_results(BINARY_SMF_RESULTS, BINARY_SMF_BINS, "../tmp/smf_related_work_binary.png")
    visualize_results(BINARY_GI_RESULTS, BINARY_GI_BINS, "../tmp/gi_related_work_binary.png")

def visualize_results(results, bins, output_path):

    rows = []
    for r in results:
        for roc, bin_name in zip(r['auc-roc'], bins):
            rows.append({
                "name" : r['name'],
                "bin" : bin_name,
                "auc-roc" : roc
            })
    df = pd.DataFrame(rows)
    
    colors = [r['color'] for r in results]

    g = sns.catplot(x="bin", 
        y="auc-roc",
        hue="name",
        data=df,
        kind="bar",
        height=10,
        aspect=2,
        palette=colors,
        edgecolor='black',
        legend=False,
        linewidth=plot_cfg["bar_border_size"],
        saturation=1)
    ax = g.ax

    for i, bar in enumerate(ax.patches):
        
        hid = i // len(bins)
        
        bar.set_hatch(results[hid].get('hatch', ''))
        
        bid = i % len(bins)

        val = results[hid]['auc-roc'][bid]

        if np.isnan(val):
            txt = ax.text(bar.xy[0] + bar.get_width()/2, 
                0.51, "Not reported", rotation=90, ha="center", va="bottom", 
                    color=colors[hid],
                    fontfamily='sans',
                    fontsize=plot_cfg['bar_label_size'], weight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='black')])
        
        bar_labels = results[hid].get('bar_labels', None)
        if bar_labels is not None:
            print(bar_labels)
            bar_label = bar_labels[bid]
            txt = ax.text(bar.xy[0] + bar.get_width()/2, 
                0.51, bar_label, rotation=90, ha="center", va="bottom", 
                    color="black",
                    fontfamily='sans',
                    fontsize=plot_cfg['bar_label_size']*0.85, weight='bold')
            #txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='black')])
        

    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel("AUC-ROC", fontsize=plot_cfg['ylabel_size'], fontweight='bold')

    if len(bins) == 1:
        ax.set_xlabel(bins[0], fontsize=plot_cfg['xlabel_size'], fontweight='bold', labelpad=20)
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("")
    # ax.set_xticklabels([])
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.set_ylim([0.5, 1.0])

    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.legend(
        bbox_to_anchor=(0, 1.05, 1, 0.102),
        frameon=False,
            fontsize=plot_cfg['legend_size'], loc="upper left", ncol=2,  mode="expand")

    plt.savefig(output_path, bbox_inches='tight', dpi=100)



if __name__ == "__main__":
    main()
