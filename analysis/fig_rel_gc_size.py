import networkx as nx 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

species = [
    {
        "path" : "../generated-data/ppc_yeast",
        "name" : "$\\it{S. cerevisiae}$",
        "full_size" : 6724,
        "color" : "#FF3AF2"
    },
    {
        "path" : "../generated-data/ppc_pombe",
        "name" : "$\\it{S. pombe}$",
        "color" : "#FFA93A"
    },
    {
        "path" : "../generated-data/ppc_human",
        "name" : "$\\it{H. sapiens}$",
        "color" : "#00c20d",
        "label_outside" : True
    },
    {
        "path" : "../generated-data/ppc_dro",
        "name" : "$\\it{D. melanogaster}$",
        "color" : "#3A90FF",
        "full_size" : 22954,
        "label_outside" : True
    }
]

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

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'

def main():

    gc_rel_sizes = []

    for s in species:

        G = nx.read_gpickle(s['path'])
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        gc = components[0]

        full_size = G.number_of_nodes()
        if 'full_size' in s:
            full_size = s['full_size']

        rel_size = len(gc) * 100 / full_size
        gc_rel_sizes.append(rel_size)
    
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    g = sns.barplot([s['name'] for s in species], gc_rel_sizes, saturation=1, 
       ax=ax, palette=[s['color'] for s in species])

    # plot labels
    for i, s in enumerate(species):
        val = gc_rel_sizes[i] / 2
        fsize = s['fsize'] if 'fsize' in s else plot_cfg['bar_label_size']
        
        if 'label_outside' in s and s['label_outside']:
            ax.text(i, gc_rel_sizes[i] * 1.05, s['name'], rotation=90, ha="center", va="bottom", fontsize=fsize, weight="bold")
        else:
            ax.text(i, val, s['name'], rotation=90, 
                ha="center", va="center", fontsize=fsize, weight='bold')

    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel('Relative GC Size %', fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 100])
    ax.set_xlabel("")
    ax.set_xticklabels([])

    plt.savefig("../figures/relative_gc.png", bbox_inches='tight', dpi=100, quality=100)
    
    plt.show()

if __name__ == "__main__":
    main()
