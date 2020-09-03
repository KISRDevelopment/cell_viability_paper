import os 
import subprocess 
import sys 
import json 
import shlex
import glob 
import os 
import numpy as np 
import matplotlib.pyplot as plt
import models.cv 
import scipy.stats as stats

COLORS = ["#FF3AF2", "#FFA93A", "#3AFF46", "#3A90FF"]
plot_cfg = {
    "x_tick_label_size" : 24,
    "y_tick_label_size" : 22,
    "xlabel_size" : 40,
    "ylabel_size" : 20,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 24
}
plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    goid_names = json.load(f)


# files = glob.glob('../results/gi_interpretation/yeast_mn/*.npz')

# # errors = []
# # for label, mu, std in zip(labels, muW, stdW):
# #     ci = stats.t.interval(0.95, W.shape[0]-1, loc=mu, scale=std)
# #     errors.append(ci)
# #     #print("%64s %8.4f (%0.3f - %0.3f)" % (label, mu, ci[0], ci[1]))
# # errors = np.array(errors)

# # f, axes = plt.subplots(1, 4, figsize=(40, 20), sharey=True, sharex=True)

# # errors[:,0] = muW - errors[:,0]
# # errors[:,1] = errors[:,1] - muW 
# # muW = muW[ix]
# # errors = errors[ix, :]
# # labels = np.array(labels)
# # labels = labels[ix]
# # ax = axes[0]
# # ax.plot([0.0, 0.0], [-0.5, W.shape[1] - 0.5], linewidth=5, color='grey', linestyle='--')
# # ax.errorbar(y = np.arange(W.shape[1]), x = muW, xerr=errors.T, fmt='o', 
# #     capthick=2, capsize=2, elinewidth=2, markersize=5, color=COLORS[0])
# # ax.set_yticks(np.arange(W.shape[1]))
# # ax.set_yticklabels(labels)
# # ax.tick_params(axis='y', labelsize=plot_cfg['tick_label_size'])
# # ax.tick_params(axis='x', labelsize=plot_cfg['tick_label_size'])
# # ax.set_xlabel('Coefficient Value', fontsize=plot_cfg['xlabel_size'], fontweight='bold')
# # plt.savefig("../tmp/gi_interpreation_coeff.png", bbox_inches='tight')


def main(paths, output_path):

    f, axes = plt.subplots(1, len(paths), figsize=(40, 30), sharey=True, sharex=True)

    ix = None 
    all_mus = []
    for i, p in enumerate(paths):

        muW, stdW, errors, labels = load_weights(p)
        
        errors[:, 0] = muW - errors[:, 0]
        errors[:, 1] = errors[:, 1] - muW 

        if ix is None:
            ix = np.argsort(muW)

        muW = muW[ix]
        errors = errors[ix, :]
        labels = labels[ix]

        all_mus.append(muW)

        ax = axes[i]

        ax.plot([0.0, 0.0], [-0.5, errors.shape[0] - 0.5], linewidth=5, color='grey', linestyle='--')
        ax.errorbar(y = np.arange(errors.shape[0]), x = muW, xerr=errors.T, fmt='o', capthick=3, capsize=6, elinewidth=3, markersize=6,  color=COLORS[i])
        ax.set_yticks(np.arange(errors.shape[0]))
        ax.set_yticklabels(labels)
        ax.tick_params(axis='y', labelsize=plot_cfg['y_tick_label_size'])
        ax.tick_params(axis='x', labelsize=plot_cfg['x_tick_label_size'])
        ax.set_xlabel('Coefficient Value', fontsize=plot_cfg['xlabel_size'], fontweight='bold')
        ax.set_ylim([-0.5, errors.shape[0] - 0.5])

    plt.savefig(output_path + ".png", bbox_inches='tight', dpi=150)
    plt.close()

    #plt.show()

    n = len(paths)

    corr_matrix = np.zeros((n-1, n))
    corr_matrix[:] = np.nan
    
    for i in range(n):
        for j in range(i+1, n):
            a = all_mus[i]
            b = all_mus[j]
            rho, _ = stats.spearmanr(a, b)
            corr_matrix[i, j] = rho 
            print("%s %s %f" % (i, j, rho))
    corr_matrix = corr_matrix[:,1:]

    print(corr_matrix)

    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(corr_matrix, cmap=plt.get_cmap('Reds'), vmin=0, vmax=1)
    for i in range(corr_matrix.shape[0]):
        for j in range(i, corr_matrix.shape[1]):
            ax.text(j, i, "%0.2f" % corr_matrix[i, j], ha="center", va="center", 
                fontsize=plot_cfg['annot_size'])

        plt.setp(ax.spines.values(), linewidth=0)
        
    xlabels = ax.get_xticks()

    ix = np.isin(xlabels, np.arange(n-1))
    xlabels = xlabels.astype(str)
    xlabels[ix] = ['$\\it{S. pombe}$', '$\\it{H. sapiens}$', '$\\it{D. melanogaster}$']
    xlabels[~ix] = ''

    ylabels = ax.get_yticks()
    ix = np.isin(ylabels, np.arange(n-1))
    ylabels = ylabels.astype(str)
    ylabels[ix] = ['$\\it{S. cerevisiae}$', '$\\it{S. pombe}$', '$\\it{H. sapiens}$']
    ylabels[~ix] = ''


    ax.xaxis.tick_top()
    ax.set_xticklabels(xlabels, fontsize=plot_cfg['annot_size'], rotation=90)
    ax.set_yticklabels(ylabels, fontsize=plot_cfg['annot_size'])
    ax.xaxis.set_tick_params(length=0, width=0, which='both')
    ax.yaxis.set_tick_params(length=0, width=0, which='both')
    plt.savefig(output_path + "_corr_matrix.png", bbox_inches='tight')

    #plt.show()

def load_weights(path):

    files = glob.glob(path + '/*.npz')

    W = []
    ref_class = 1
    labels = None
    for file in files:
        d = np.load(file)

        weights = d['weights']
        biases = d['biases']

        new_biases = biases - biases[ref_class]
        new_weights = weights - np.expand_dims(weights[:, ref_class], 1)

        W.append(new_weights)
        labels = d['labels']

    W = np.array(W)
    W = W[:,:,0]

    labels = [process_label(l) for l in labels]
    labels = [l[0].upper() + l[1:] for l in labels]

    muW =  np.mean(W, axis=0)
    stdW = stats.sem(W, axis=0)

    errors = []
    for mu, std in zip(muW, stdW):
        ci = stats.t.interval(0.95, W.shape[0]-1, loc=mu, scale=std)
        errors.append(ci)
    errors = np.array(errors)

    return muW, stdW, errors, np.array(labels)

smf_lookup = {
    'smf_0.00.0' : 'Lethal/Lethal',
    'smf_0.01.0' : 'Lethal/Reduced Growth',
    'smf_0.02.0' : 'Lethal/Normal',
    'smf_1.01.0' : 'Reduced Growth/Reduced Growth',
    'smf_1.02.0' : 'Reduced Growth/Normal',
    'smf_2.02.0' : 'Normal/Normal',
    'smf_00' : 'Lethal/Lethal',
    'smf_01' : 'Lethal/Reduced Growth',
    'smf_02' : 'Lethal/Normal',
    'smf_11' : 'Reduced Growth/Reduced Growth',
    'smf_12' : 'Reduced Growth/Normal',
    'smf_22' : 'Normal/Normal',
}
def process_label(lbl):

    if lbl.startswith('sgo_both_'):
        
        goid = lbl.replace('sgo_both_', '')
        go_name = goid_names[goid]
        lbl = '%s (both)' % go_name
    elif lbl.startswith('sgo_either_'):
        goid = lbl.replace('sgo_either_', '')
        go_name = goid_names[goid]
        lbl = '%s (either)' % go_name
    elif lbl == 'sum_lid':
        lbl = 'LID (sum)'
    elif lbl.startswith('smf_'):
        lbl = smf_lookup[lbl]
    elif lbl == 'spl':
        lbl = 'Shortest Path Length'

    return lbl 

if __name__ == "__main__":
    output_path = sys.argv[1]
    paths = sys.argv[2:]
    main(paths, output_path)