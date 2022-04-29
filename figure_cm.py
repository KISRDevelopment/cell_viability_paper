import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import matplotlib 

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

def load_results(path):
    with open(path, 'r') as f:
        results = json.load(f)['results']
    
    return sorted(results, key=lambda r: r['split_id'])

def plot_cm(results_path, color, classes, output_path):

    results = load_results(results_path)

    cms = []
    for r in results:
        cm = np.array(r['cm'])
        cm = cm / np.sum(cm, axis=1, keepdims=True)
        cms.append(cm)
    cm = np.mean(cms, axis=0)


    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["white",color])

    f, ax = plt.subplots(figsize=(10, 10))

    displayed_cm = np.zeros((plot_cfg['max_cm_classes'], plot_cfg['max_cm_classes']))
    displayed_cm[:cm.shape[0], :cm.shape[0]] = cm 

    ax.imshow(displayed_cm, cmap=cmap)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, "%0.2f" % cm[i, j], ha="center", va="center", 
            fontsize=plot_cfg['annot_size'])

    xlabels = ax.get_xticks()
    ix = np.isin(xlabels, np.arange(len(classes)))
    xlabels = xlabels.astype(str)
    xlabels[ix] = classes
    xlabels[~ix] = ''

    ax.set_xticklabels(xlabels, fontsize=plot_cfg['annot_size'])
    ax.set_yticklabels(xlabels, fontsize=plot_cfg['annot_size'])
        
    ax.xaxis.set_tick_params(length=0, width=0, which='both', colors=color)
    ax.yaxis.set_tick_params(length=0, width=0, which='both', colors=color)
    ax.xaxis.tick_top()
    plt.setp(ax.spines.values(), linewidth=0)

    plt.savefig(output_path, bbox_inches='tight')

if __name__ == "__main__":

    results_path = sys.argv[1]
    color = sys.argv[2]
    if sys.argv[3] == 'smf':
        classes = ['L', 'R', 'N']
    elif sys.argv[3] == 'binary_smf':
        classes = ['L', 'V']
    elif sys.argv[3] == 'gi':
        classes = ['-', 'N', '+', 'S']
    elif sys.argv[3] == 'binary_gi':
        classes = ['I', 'N']
    output_path = sys.argv[4]

    plot_cm(results_path, color, classes, output_path)

