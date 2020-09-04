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

plot_cfg = {
    "x_tick_label_size" : 32,
    "y_tick_label_size" : 32,
    "xlabel_size" : 40,
    "ylabel_size" : 40,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 24,
    "title_size" : 50
}
errorbar_props = {
    "capthick" : 5, 
    "capsize" : 10, 
    "elinewidth" : 5,
    "markersize" : 10,
    "fmt" : "o"  
}

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    goid_names = json.load(f)

def main(cfg):

    plot_cfg.update(cfg['plot_cfg'])

    f, axes = plt.subplots(len(cfg['spec']), 1, figsize=(30, 20), sharey=True, sharex=True)

    ix = None
    all_mus = []
    for i, s in enumerate(cfg['spec']):
        if cfg['model'] == 'mn':
            muW, stdW, errors, labels = load_weights_mn(s['path'], cfg['ref_class'])
            
            c = cfg["target_class"]
            c_muW = muW[:, c]
            c_errors = errors[:, :, c]

        else:
            c_muW, _, c_errors, labels = load_weights_orm(s['path'])
        
        c_errors[:, 0] = c_muW - c_errors[:, 0]
        c_errors[:, 1] = c_errors[:, 1] - c_muW 


        for j in range(len(labels)):
            print("%64s %8.4f" % (labels[j], c_muW[j]))
        print()
        if ix is None:
            ix = np.argsort(c_muW)

        c_muW = c_muW[ix]
        c_errors = c_errors[ix, :]
        labels = labels[ix]

        all_mus.append(c_muW)

        ax = axes[len(cfg['spec'])-i-1]

        ax.plot([-0.5, c_errors.shape[0] - 0.5], [0.0, 0.0], linewidth=5, 
            color='grey', linestyle='--')
        ax.errorbar(x = np.arange(c_errors.shape[0]), y = c_muW, yerr=c_errors.T, 
            color=s["color"], **errorbar_props)
        ax.set_xticks(np.arange(c_errors.shape[0]))
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', labelsize=plot_cfg['x_tick_label_size'], rotation=90)
        ax.tick_params(axis='y', labelsize=plot_cfg['y_tick_label_size'])
        ax.set_ylabel('Coefficient\nValue', fontsize=plot_cfg['ylabel_size'], fontweight='bold')
        ax.set_xlim([-0.5, c_errors.shape[0] - 0.5])
        ax.grid()
        
    plt.savefig(cfg['output_path'], bbox_inches='tight', dpi=150)

    n = len(cfg['spec'])

    corr_matrix = np.zeros((n, n))
    corr_matrix[:] = np.nan
    
    for i in range(n):
        for j in range(i+1, n):
            a = all_mus[i]
            b = all_mus[j]
            rho, _ = stats.spearmanr(a, b)
            corr_matrix[i, j] = rho 
            
    
    print(corr_matrix)

def load_weights_orm(path):
    files = glob.glob(path + '/*.npz')

    W = []
    labels = None
    for file in files:
        d = np.load(file)
        W.append(d['weights'])
        labels = d['labels']

    W = np.array(W).squeeze()

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

def load_weights_mn(path, ref_class):

    files = glob.glob(path + '/*.npz')

    W = []
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

    if lbl in goid_names:
        return goid_names[lbl]
    elif lbl == 'lid':
        return 'LID'
    elif lbl == 'pident':
        return 'Percent Identity'
    elif lbl.startswith('sgo_both_'):
        
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
    path = sys.argv[1]
    with open(path, 'r') as f:
        cfg = json.load(f)

    main(cfg)