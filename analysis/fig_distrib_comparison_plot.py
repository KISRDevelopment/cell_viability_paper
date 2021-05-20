import numpy as np 
import matplotlib.colors 
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import utils.eval_funcs as eval_funcs
import pandas as pd


plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
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
def plot_distrib(df, pallete, xlabel, ylabel, max_x_categories, ylim, stars_color, output_path, xlim=None, legend_pos=(1, 1.05), stars_pos=(0.53, 0.98)):
    distinct_xs = set(df['x'])
    distinct_hues = set(df['hue'])

    dummy_rows = []

    if max_x_categories > 0:
        for i in range(max_x_categories - len(distinct_xs)):
            for hue in distinct_hues:
                dummy_rows.append({
                    "x" : " " * (i+1),
                    "y" : 0,
                    "hue" : hue
                })
    final_df = pd.concat((df, pd.DataFrame(dummy_rows)))

    f, ax = plt.subplots(1, 1, figsize=(15, 10))
    g = sns.barplot(x="x", 
                    y="y", 
                    hue="hue",
                    ax=ax,
                    data=final_df, 
                    palette=pallete)
    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'], pad=15)
    ax.set_ylabel(ylabel, fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=plot_cfg['xlabel_size'], fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(),linewidth=plot_cfg["border_size"], color='black')
    if ylim != None:
        ax.set_ylim(ylim)
    if xlim != None:
        ax.set_xlim(xlim)

    # construct R
    xs = list(distinct_xs)
    hues = list(distinct_hues)
    R = np.zeros((len(xs), len(hues)))
    for i in range(len(xs)):
        for j in range(len(hues)):
            R[i, j] = df[(df['x'] == xs[i]) & (df['hue'] == hues[j])]['raw_y']
    chisq, p, ddof = eval_funcs.chi2(R)
    stars = '*' * eval_funcs.compute_stars(p, 0.05)
    ax.legend(frameon=False, fontsize=plot_cfg['legend_size'], loc='upper right', bbox_to_anchor=legend_pos, ncol=1)
    ax.text(stars_pos[0], stars_pos[1], '****', 
        transform=ax.transAxes,
        color=stars_color, 
        ha="center", 
        va="top",
        weight='bold', 
        fontsize=plot_cfg['stars_label_size'])
    
    plt.savefig(output_path, bbox_inches='tight', dpi=100)