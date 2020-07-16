import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
import json 
import sys


plot_cfg = {
    "tick_label_size" : 50,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 82,
    "legend_size" : 42
}

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'
def main():
    path = sys.argv[1]

    with open(path, 'r') as f:
        cfg = json.load(f)
    
    species, classes = cfg['species'], cfg['classes']

    rows = []

    for s in species:
        d = np.load(s['path'])
        y = d['y']

        n_bins = int(np.max(y) + 1)

        counts = np.array([np.sum(y == b) for b in range(n_bins)])
        rcounts = counts * 100 / np.sum(counts)

        species_classes = s['classes'] if 'classes' in s else classes 

        for i, c in enumerate(species_classes):
            rows.append({
                "species" : s['name'],
                "class" : c,
                "count" : counts[i],
                "rel_count" : rcounts[i]
            })
    
    df = pd.DataFrame(rows)
    
    visualize(df, cfg, 'rel_count', "../tmp/%s_rel.png" % cfg['output_name'], False)
    visualize(df, cfg, 'count', "../tmp/%s_raw.png" % cfg['output_name'], True)

def visualize(df, cfg, y, output_path, logy):
    species = cfg['species']

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    bar = sns.barplot(x="class", y=y, hue="species", 
        data=df, ax=ax, palette=[s['color'] for s in species], saturation=1)    
    
    print(df)
    for i, b in enumerate(bar.patches):
        
        # get the species of that bar
        sid = i // len(cfg['classes'])
        group = i % len(cfg['classes'])

        s = species[sid]
        
        if group == 1 and 'border_only' in s and s['border_only']:
            b.set_color('white')

            w = b.get_width()
            
            b.set_width(w*0.85)
            b.set_edgecolor(s['color'])
            b.set_linewidth(10)
            

    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])

    ylabel = '% of Observations' if y=='rel_count' else '# of Observations'
    ax.set_ylabel(ylabel, fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if y == 'rel_count':
        ax.set_ylim([0, 100])
    ax.set_xlabel("")
    ax.legend().remove()
    ax.legend(
        bbox_to_anchor=(0, 1.25, 1, 0.102),
        fontsize=plot_cfg['legend_size'], loc="upper left", ncol=2,  mode="expand")

    if logy:
        ax.set_yscale('log')
    plt.savefig(output_path, bbox_inches='tight', dpi=100, quality=100)
    
    plt.show()

if __name__ == "__main__":
    main()
