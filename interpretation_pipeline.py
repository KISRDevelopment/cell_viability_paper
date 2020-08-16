import os 
import subprocess 
import sys 
import json 
import shlex
import glob 
import os 
import numpy as np 
import matplotlib.pyplot as plt

import models.smf_ordinal
import models.cv 
import scipy.stats as stats

plot_cfg = {
    "tick_label_size" : 24,
    "xlabel_size" : 40,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 24
}
plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'

# SMF feature interpretation 

# models.cv.main("models.smf_ordinal", "cfgs/models/yeast_smf_orm.json", 
#     "../results/smf_interpretation/yeast_orm", interpreation=True, num_processes=10, epochs=500)

models.cv.main("models.smf_ordinal", "cfgs/models/pombe_smf_orm.json", 
    "../results/smf_interpretation/pombe_orm", interpreation=True, num_processes=10, epochs=500)

models.cv.main("models.smf_ordinal", "cfgs/models/human_smf_orm.json", 
    "../results/smf_interpretation/human_orm", interpreation=True, num_processes=10, epochs=500)

models.cv.main("models.smf_ordinal", "cfgs/models/dro_smf_orm.json", 
    "../results/smf_interpretation/dro_orm", interpreation=True, num_processes=10, epochs=500)

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    gene_ids_to_names = json.load(f)

files = glob.glob('../results/smf_interpretation/yeast_orm/*.npz')

label_rewrites = {
    "lid" : "LID",
    "pident" : "Percent Identity"
}


COLORS = ["#FF3AF2", "#FFA93A", "#3AFF46", "#3A90FF"]
W = []
labels = None
for file in files:
    d = np.load(file)
    W.append(d['weights'])
    labels = d['labels']

labels = [gene_ids_to_names[l] if l in gene_ids_to_names else l for l in labels]
labels = [label_rewrites.get(l, l) for l in labels]
labels = [l[0].upper() + l[1:] for l in labels]
W = np.array(W).squeeze()

muW =  np.mean(W, axis=0)
stdW = stats.sem(W, axis=0)
ix = np.argsort(muW)

errors = []
for label, mu, std in zip(labels, muW, stdW):
    ci = stats.t.interval(0.95, W.shape[0]-1, loc=mu, scale=std)
    errors.append(ci)
    #print("%64s %8.4f (%0.3f - %0.3f)" % (label, mu, ci[0], ci[1]))
errors = np.array(errors)

f, axes = plt.subplots(1, 4, figsize=(40, 20), sharey=True, sharex=True)

errors[:,0] = muW - errors[:,0]
errors[:,1] = errors[:,1] - muW 
muW = muW[ix]
errors = errors[ix, :]
labels = np.array(labels)
labels = labels[ix]
ax = axes[0]
ax.plot([0.0, 0.0], [-0.5, W.shape[1] - 0.5], linewidth=5, color='grey', linestyle='--')
ax.errorbar(y = np.arange(W.shape[1]), x = muW, xerr=errors.T, fmt='o', 
    capthick=5, capsize=10, elinewidth=5, markersize=10, color=COLORS[0])
ax.set_yticks(np.arange(W.shape[1]))
ax.set_yticklabels(labels)
ax.tick_params(axis='y', labelsize=plot_cfg['tick_label_size'])
ax.tick_params(axis='x', labelsize=plot_cfg['tick_label_size'])
ax.set_xlabel('Coefficient Value', fontsize=plot_cfg['xlabel_size'], fontweight='bold')

all_mus = [muW]

for i, species in enumerate(['pombe', 'human', 'dro']):

    files = glob.glob('../results/smf_interpretation/%s_orm/*.npz' % species)
    W = []
    labels = None
    for file in files:
        d = np.load(file)
        W.append(d['weights'])
        labels = d['labels']

    W = np.array(W).squeeze()

    muW =  np.mean(W, axis=0)
    stdW = stats.sem(W, axis=0)
    
    errors = []
    for label, mu, std in zip(labels, muW, stdW):
        ci = stats.t.interval(0.95, W.shape[0]-1, loc=mu, scale=std)
        errors.append(ci)
    errors = np.array(errors)

    errors[:,0] = muW - errors[:,0]
    errors[:,1] = errors[:,1] - muW 
    muW = muW[ix]
    errors = errors[ix, :]
    labels = np.array(labels)
    labels = labels[ix]

    all_mus.append(muW)

    ax = axes[i+1]
    ax.plot([0.0, 0.0], [-0.5, W.shape[1] - 0.5], linewidth=5, color='grey', linestyle='--')
    ax.errorbar(y = np.arange(W.shape[1]), x = muW, xerr=errors.T, fmt='o', 
        capthick=5, capsize=10, elinewidth=5, markersize=10, color=COLORS[i+1])
    ax.set_yticks(np.arange(W.shape[1]))
    ax.tick_params(axis='y', labelsize=plot_cfg['tick_label_size'])
    ax.tick_params(axis='x', labelsize=plot_cfg['tick_label_size'])
    ax.set_xlabel('Coefficient Value', fontsize=plot_cfg['xlabel_size'], fontweight='bold')

#plt.show()
plt.savefig("../tmp/smf_interpreation_coeff.png", bbox_inches='tight')


corr_matrix = np.zeros((3, 4))
corr_matrix[:] = np.nan
 
for i in range(4):
    for j in range(i+1, 4):
        a = all_mus[i]
        b = all_mus[j]
        rho, _ = stats.spearmanr(a, b)
        corr_matrix[i, j] = rho 

corr_matrix = corr_matrix[:,1:]


f, ax = plt.subplots(1, 1, figsize=(10, 10))


ax.imshow(corr_matrix, cmap=plt.get_cmap('Reds'), vmin=0, vmax=1)
for i in range(corr_matrix.shape[0]):
    for j in range(i, corr_matrix.shape[1]):
        ax.text(j, i, "%0.2f" % corr_matrix[i, j], ha="center", va="center", 
            fontsize=plot_cfg['annot_size'])

    plt.setp(ax.spines.values(), linewidth=0)
    
xlabels = ax.get_xticks()

ix = np.isin(xlabels, np.arange(3))
xlabels = xlabels.astype(str)
xlabels[ix] = ['$\\it{S. pombe}$', '$\\it{H. sapiens}$', '$\\it{D. melanogaster}$']
xlabels[~ix] = ''

ylabels = ax.get_yticks()
ix = np.isin(ylabels, np.arange(3))
ylabels = ylabels.astype(str)
ylabels[ix] = ['$\\it{S. cerevisiae}$', '$\\it{S. pombe}$', '$\\it{H. sapiens}$']
ylabels[~ix] = ''


ax.xaxis.tick_top()
ax.set_xticklabels(xlabels, fontsize=plot_cfg['annot_size'], rotation=90)
ax.set_yticklabels(ylabels, fontsize=plot_cfg['annot_size'])
ax.xaxis.set_tick_params(length=0, width=0, which='both')
ax.yaxis.set_tick_params(length=0, width=0, which='both')

plt.savefig("../tmp/smf_interpreation_corr.png", bbox_inches='tight')

plt.show()