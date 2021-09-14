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
import pandas as pd 
import numpy.random as rng 
import sklearn.metrics 
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors 

plot_cfg = {
    "x_tick_label_size" : 58,
    "y_tick_label_size" : 40,
    "xlabel_size" : 40,
    "ylabel_size" : 55,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 70,
    "title_size" : 50
}
errorbar_props = {
    "capthick" : 5, 
    "capsize" : 10, 
    "elinewidth" : 10,
    "markersize" : 20,
    "fmt" : "o"  
}

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    goid_names = json.load(f)

def main(cfg):

    plot_cfg.update(cfg['plot_cfg'])

    n_subplots = cfg.get('subplots', len(cfg['spec']))
    f, axes = plt.subplots(n_subplots, 1, figsize=plot_cfg.get('figsize', (50, 20)), sharey=True, sharex=True)

    ix = None
    all_mus = []
    all_stds = []
    all_errors = []

    output_dfs = []

    for i, s in enumerate(cfg['spec']):
        
        if cfg['model'] == 'mn':
            muW, stdW, errors, labels = load_weights_mn(s['path'], 
                s.get('ref_class', cfg['ref_class']))
            
            print(labels)
            
            c = s.get('target_class', cfg["target_class"])

            c_muW = muW[:, c]
            c_errors = errors[:, :, c]
            stdW = stdW[:, c]
            
        else:
            c_muW, stdW, c_errors, labels = load_weights_orm(s['path'])
        
        c_errors[:, 0] = c_muW - c_errors[:, 0]
        c_errors[:, 1] = c_errors[:, 1] - c_muW 

        if ix is None:
            ix = np.argsort(c_muW)


        c_muW = c_muW[ix]
        c_errors = c_errors[ix, :]
        labels = labels[ix]

        # for j in range(len(labels)):
        #     print("%64s %8.4f (%8.4f - %8.4f)" % (labels[j], c_muW[j], c_muW[j] - c_errors[j,0], c_muW[j] + c_errors[j,1]))
        # print()
        n = s['text_name']
        df_dict = {
            "%s_Mean Coefficient Value" % n : c_muW,
            "%s_95%% CI Lower" % n : c_muW - c_errors[:, 0],
            "%s_95%% CI Upper" % n : c_muW + c_errors[:, 1],
            "%s_Exp Mean Coefficient Value" % n : np.exp(c_muW)
        }
        if i == 0:
            df_dict['Feature'] = labels 
        
        output_df = pd.DataFrame(df_dict)
        output_dfs.append(output_df)

        #output_df.to_excel(writer, index=False, sheet_name=s['text_name'])

        all_mus.append(c_muW)
        all_stds.append(stdW[ix])
        all_errors.append(c_errors)

        ax = axes[n_subplots-i-1]
        n_entries = cfg.get('n_entries', c_errors.shape[0])

        ax.plot([-0.5, n_entries - 0.5], [0.0, 0.0], linewidth=5, 
            color='grey', linestyle='--')
        ax.errorbar(x = np.arange(c_errors.shape[0]), y = c_muW, yerr=c_errors.T, 
            color=s["color"], **errorbar_props)

        
        ax.set_xticks(np.arange(n_entries))

        print("Entries: %d" % c_errors.shape[0])
        n_to_add = n_entries - len(labels)
        labels = labels.tolist() + ([""] * n_to_add)

        ax.set_xticklabels(labels)

        ax.tick_params(axis='x', labelsize=plot_cfg['x_tick_label_size'], rotation=90)
        ax.tick_params(axis='y', labelsize=plot_cfg['y_tick_label_size'])
        if cfg.get('ylabel', True):
            ax.set_ylabel('Coefficient\nValue', fontsize=plot_cfg['ylabel_size'], fontweight='bold')
        ax.set_xlim([-0.5, n_entries - 0.5])
        if cfg.get('ylim', None) is not None:
            ax.set_ylim(cfg['ylim'])
        ax.grid()
    
    
    output_df = pd.concat(output_dfs, axis=1).set_index('Feature')
    output_df.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in output_df.columns])
    output_df.to_excel(cfg['output_path']+'.xlsx', sheet_name='Sheet1', index=True)

    f.subplots_adjust(hspace=0.05)

    plt.savefig(cfg['output_path'], bbox_inches='tight', dpi=150)

    # spearman corr
    n = len(cfg['spec'])
    corr_matrix = np.zeros((n, n))
    corr_matrix[:] = np.nan
    pval_matrix = np.zeros((n, n))
    pval_matrix[:] = np.nan
    for i in range(n):
        for j in range(i+1, n):
            a = all_mus[i]
            b = all_mus[j]
            rho, pval = stats.spearmanr(a, b)
            corr_matrix[i, j] = rho 
            pval_matrix[i,j] = pval
    visualize_corr_matrix(corr_matrix, cfg, cfg['output_path'] + '_spearman_corr.png')
    print(pval_matrix)
    # kappa
    corr_matrix = np.zeros((n, n))
    corr_matrix[:] = np.nan
    n_sim = 100
    for i in range(len(cfg['spec'])):
        mu = all_mus[i]
        lower = mu - all_errors[i][:,0]
        upper = mu + all_errors[i][:,1]

        for j in range(i+1, len(cfg['spec'])):
            dsts = []
            for s in range(n_sim):
                
                a = stats.norm.rvs(loc=all_mus[i], scale=all_stds[i]) > 0.0
                b = stats.norm.rvs(loc=all_mus[j], scale=all_stds[j]) > 0.0
                
                dst = sklearn.metrics.cohen_kappa_score(a, b)
                
                dsts.append(dst)
                
            corr_matrix[i, j] = np.mean(dsts)
    visualize_corr_matrix(corr_matrix, cfg, cfg['output_path'] + '_cohen_kappa_corr.png')



def visualize_corr_matrix(corr_matrix, cfg, output_path):
    n = corr_matrix.shape[0]
    
    # drop the first column and last row
    corr_matrix = corr_matrix[:, 1:]
    corr_matrix = corr_matrix[:-1, :]
    
    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    cmap = LinearSegmentedColormap.from_list('mycmap', ['white', cfg.get('color', 'red')])

    ax.imshow(corr_matrix, cmap=cmap, vmin=0, vmax=1)
    for i in range(corr_matrix.shape[0]):
        for j in range(i, corr_matrix.shape[1]):
            ax.text(j, i, "%0.2f" % corr_matrix[i, j], 
                ha="center", va="center", 
                fontsize=plot_cfg['annot_size'])
        plt.setp(ax.spines.values(), linewidth=0)
    

    xlabels = ax.get_xticks()
    ix = np.isin(xlabels, np.arange(n-1))
    xlabels = xlabels.astype(str)
    xlabels[ix] = [e["name"] for e in cfg["spec"]][1:]
    xlabels[~ix] = ''

    ylabels = ax.get_xticks()
    ix = np.isin(ylabels, np.arange(n-1))
    ylabels = ylabels.astype(str)
    ylabels[ix] = [e["name"] for e in cfg["spec"]][:-1]
    ylabels[~ix] = ''

    ax.xaxis.tick_top()
    ax.set_xticklabels(xlabels, fontsize=plot_cfg['annot_size'], rotation=90)
    ax.set_yticklabels(ylabels, fontsize=plot_cfg['annot_size'])
    ax.xaxis.set_tick_params(length=0, width=0, which='both', pad=10)
    ax.yaxis.set_tick_params(length=0, width=0, which='both', pad=10)
    
    plt.savefig(output_path, bbox_inches='tight')

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

    'smf_00' : 'Lethal/Lethal',
    'smf_01' : 'Lethal/Reduced Growth',
    'smf_02' : 'Lethal/Normal',
    'smf_11' : 'Reduced Growth/Reduced Growth',
    'smf_12' : 'Reduced Growth/Normal',
    'smf_22' : 'Normal/Normal',

    'smf_0.00.00.0' : 'Lethal/Lethal/Lethal',
    'smf_0.00.01.0' : 'Lethal/Lethal/Reduced Growth',
    'smf_0.00.02.0' : 'Lethal/Lethal/Normal',
    'smf_0.01.01.0' : 'Lethal/Reduced Growth/Reduced Growth',
    'smf_0.01.02.0' : 'Lethal/Reduced Growth/Normal',
    'smf_0.02.02.0' : 'Lethal/Normal/Normal',
    'smf_1.01.01.0' : 'Reduced Growth/Reduced Growth/Reduced Growth',
    'smf_1.01.02.0' : 'Reduced Growth/Reduced Growth/Normal',
    'smf_1.02.02.0' : 'Reduced Growth/Normal/Normal',
    'smf_2.02.02.0' : 'Normal/Normal/Normal'
    
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
        goid = lbl.replace('sgo_either_', '').replace('_xor','')
        go_name = goid_names[goid]
        lbl = '%s (either)' % go_name
    elif lbl.startswith('sgo_sum_'):
        goid = lbl.replace('sgo_sum_', '')
        go_name = goid_names[goid]
        lbl = '%s (Sum)' % go_name
    elif lbl == 'sum_lid':
        lbl = 'LID (sum)'
    elif lbl.startswith('smf_'):
        lbl = smf_lookup.get(lbl,lbl)
        if lbl not in smf_lookup:
            print("'%s' : ''" % lbl)
    elif lbl == 'spl':
        lbl = 'Shortest Path Length'

    return lbl 


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, 'r') as f:
        cfg = json.load(f)

    main(cfg)