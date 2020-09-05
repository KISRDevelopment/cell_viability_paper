import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import scipy.stats
import seaborn as sns
import sklearn.metrics as metrics
import sys
import json
from scipy import interp
import os 
from seaborn.utils import remove_na
import matplotlib.patches as mpatches
import utils.eval_funcs 

plot_cfg = {
    "tick_label_size" : 45,
    "xlabel_size" : 50,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 45,
    "annot_size" : 82,
    "legend_label_size" : 42
}

ALPHA = 0.05

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"

BIN_LABELS = ['Negative', 'Neutral', 'Positive', 'Suppression']

col_names = ['no. neg', 'no. neut', 'no. pos', 'no. sup']

COLORS = ['#FF0000', 'orange', '#00CC00', '#3d77ff']

def main(path, output_path):
    
    df = pd.read_excel(path, sheet_name='normalized', header=8)

    df = df[~np.isnan(df['size'])]
    rows = []
    for i, r in df.iterrows():
        for bin in range(len(BIN_LABELS)):
            col_name = col_names[bin]
            rows.append({
                "bin" : BIN_LABELS[bin],
                "prop" : 'within',
                "bin_id" : bin,
                "count" : r[col_name + ' within'] * 100
            })
            rows.append({
                "bin" : BIN_LABELS[bin],
                "prop" : 'across',
                "bin_id" : bin,
                "count" : r[col_name + ' across'] * 100
            })
    sdf = pd.DataFrame(rows)
    plot_hist(sdf, COLORS, (15, 10), output_path)

def plot_hist(sdf, colors, figsize, output_path):
    f, ax = plt.subplots(1, 1, figsize=figsize)

    bin_ids = np.unique(sdf['bin_id'])
    
    bar = sns.barplot(x="bin", 
        y="count", 
        hue="prop",
        data=sdf, 
        ci="sd",
        edgecolor='black',
        errwidth=5,
        errcolor='black',
        linewidth=plot_cfg["bar_border_size"],
        saturation=1)
    ax.legend().remove()

    across_patch = mpatches.Patch(facecolor='white', fill=False, hatch='x',label='Across')
    within_patch = mpatches.Patch(facecolor='white', edgecolor='black', fill=True, label='Within')
    
    ax.legend(handles=[within_patch, across_patch], fontsize=plot_cfg['legend_label_size'])

    ylim = ax.get_ylim()
    for i, b in enumerate(bar.patches):
        
        # get the hue of that bar
        hid = i // len(bin_ids)
        gid = i % len(bin_ids)

        group = bin_ids[gid]
        print("Group ", group)

        x,y = b.get_xy()
        height = b.get_height()
        width = b.get_width()

        if hid > 0:
            b.set_hatch('x')


            b.set_linewidth(0)


            b.set_facecolor(colors[group])

            b.set_alpha(0.5)

            

            
        else:
            bin_df = sdf[sdf['bin_id'] == group]
            within_counts = bin_df[bin_df['prop'] == 'within']['count']
            across_counts = bin_df[bin_df['prop'] == 'across']['count']
            
            statistic, pvalue = scipy.stats.ttest_rel(within_counts, across_counts)
            stars = '*' * utils.eval_funcs.compute_stars(pvalue, ALPHA)
            
            offset = ylim[1] / 15.0
            ax.text(x + width/2, 
                y + height + offset, stars, 
                color=colors[group], 
                ha="center", 
                va="top", 
                weight='bold', 
                fontsize=plot_cfg['stars_label_size'])

            b.set_color(colors[group])

    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel("%", fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.set_xlabel("")
    ax.yaxis.set_tick_params(length=10, width=1, which='both')

    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(output_path, bbox_inches='tight', dpi=100)

    plt.show()

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

catplotter = sns.categorical._CategoricalStatPlotter 
catplotter.estimate_statistic = estimate_statistic

if __name__ == "__main__":
    main()

