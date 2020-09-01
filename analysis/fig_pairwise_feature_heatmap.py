import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.metrics
import matplotlib.pyplot as plt 
import json
from collections import defaultdict
import matplotlib.colors
import matplotlib.ticker as ticker
import scipy.stats as stats
import numpy.random as rng 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys 

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"

plot_cfg = {
    "tick_label_size" : 50,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 82,
    "title_size" : 60
}

BINS = ["Neutral", "Interacting"]

def main(task_path, feature_path, feature_name, label, output_path):

    df = pd.read_csv(task_path)
    
    d = np.load(feature_path)
    F = d['F']
    F = (F + d['mu']) * d['std']
    selected_feature_ix = d['feature_labels'] == feature_name
    f = F[:,selected_feature_ix]
    
    # 0 : neutral, 1 : interacting
    df['bin'] = (df['bin'] != 1).astype(int)

    bins = sorted(np.unique(df['bin']))

    fig, axes = plt.subplots(1, len(bins), figsize=(20, 10))

    kdes = []
    for i, b in enumerate(bins):
        
        sdf = df[df['bin'] == b]
        if sdf.shape[0] >= 100000:
            sdf = sdf.sample(100000)

        PFx = f[sdf['a_id'], :]
        PFy = f[sdf['b_id'], :]

        xmin = np.min(PFx)
        xmax = np.max(PFx)

        ymin, ymax = xmin, xmax 


        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([PFx.T, PFy.T])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)

        ax = axes[i]


        ax.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
        disp_max = 15
        ax.set_xlim([xmin, disp_max])
        ax.set_ylim([xmin, disp_max])

        ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
        ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
        if i == 0:
            ax.set_ylabel(label + " A", fontsize=plot_cfg['ylabel_size'], weight='bold')
        ax.set_xlabel(label + " B", fontsize=plot_cfg['ylabel_size'], weight='bold')
        ax.set_title(BINS[i], fontsize=plot_cfg['title_size'], weight='bold')

        kdes.append(kernel)
    

    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    

    plt.show()

    ref_kernel = kdes[0]
    test_kernel = kdes[1]

    sdf = df[df['bin'] == 1]

    PFx = f[sdf['a_id'], :]
    PFy = f[sdf['b_id'], :]
    values = np.vstack([PFx.T, PFy.T])

    under_ref_kernel = np.sum(ref_kernel.logpdf(values))
    under_test_kernel = np.sum(test_kernel.logpdf(values))

    diff = under_test_kernel - under_ref_kernel
    
    print("Ref kernel logprob: %f" % under_ref_kernel)
    print("Test kernel logprob: %f" % under_test_kernel)
    print("Test/ref diff: %f" % diff)

    print("Ref integral over [0,15]: %f" % ref_kernel.integrate_box([-1, -1], [1, 1]))
    print("Test integral over [0,15]: %f" % test_kernel.integrate_box([-1, -1], [1, 1]))
    
if __name__ == "__main__":
    task_path = sys.argv[1]
    feature_path = sys.argv[2]
    feature_name = sys.argv[3]
    feature_label = sys.argv[4]
    output_path = sys.argv[5]

    main(task_path, feature_path, feature_name, feature_label, output_path)
