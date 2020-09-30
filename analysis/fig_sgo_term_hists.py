import numpy as np 
import pandas as pd 
import sys 
import networkx as nx 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats
import matplotlib.colors
import utils.eval_funcs as eval_funcs

yeast_cfg = {
    "smf_path" : "../generated-data/task_yeast_smf_30",
    "binned_smf_path" : "../generated-data/features/ppc_yeast_smf_binned.npz",
    "gpath" : "../generated-data/ppc_yeast",
    "gi_path" : "../generated-data/task_yeast_gi_hybrid",
    "sgo_path" : "../generated-data/features/ppc_yeast_common_sgo.npz",
    "text_name" : "S. cerevisiae",
    "name" : "$\\it{S. cerevisiae}$",
    "color" : "#FF3AF2",
    "label_outside" : True,
    
    "gi_label_outside" : True
}

pombe_cfg = {
    "smf_path" : "../generated-data/task_pombe_smf",
    "binned_smf_path" : "../generated-data/features/ppc_pombe_smf_binned.npz",
    "gpath" : "../generated-data/ppc_pombe",
    "gi_path" : "../generated-data/task_pombe_gi",
    "sgo_path" : "../generated-data/features/ppc_pombe_common_sgo.npz",
    "text_name" : "S. pombe",
    "name" : "$\\it{S. pombe}$",
    "color" : "#FFA93A",
    "label_outside" : True,
    
    "gi_label_outside" : True
}

human_cfg = {
    "smf_path" : "../generated-data/task_human_smf",
    "binned_smf_path" : "../generated-data/features/ppc_human_smf_binned.npz",
    "gpath" : "../generated-data/ppc_human",
    "gi_path" : "../generated-data/task_human_gi",
    "sgo_path" : "../generated-data/features/ppc_human_common_sgo.npz",
    "text_name" : "H. sapiens",
    "name" : "$\\it{H. sapiens}$",
    "color" : "#00c20d",
    
    "gi_label_outside" : True
}

dro_cfg = {
    "smf_path" : "../generated-data/task_dro_smf",
    "binned_smf_path" : "../generated-data/features/ppc_dro_smf_binned.npz",
    "gpath" : "../generated-data/ppc_dro",
    "gi_path" : "../generated-data/task_dro_gi",
    "sgo_path" : "../generated-data/features/ppc_dro_common_sgo.npz",
    "text_name" : "D. melanogaster",
    "name" : "$\\it{D. melanogaster}$",
    "color" : "#3A90FF",
    "fsize" : 38,
    "gi_fsize" : 46

}

ALPHA = 0.05

SMF_BINS = ['L', 'R', 'N']
GI_BINS = ['I', 'N']
cfgs = [yeast_cfg, pombe_cfg, human_cfg, dro_cfg]

plot_cfg = {
    "tick_label_size" : 50,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 48,
    "stars_label_size" : 48,
    "annot_size" : 82,
    "legend_size" : 38
}
plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['mathtext.fontset'] = 'stix'
np.set_printoptions(precision=2)
def main(cfgs, output_path):

    # prepare data
    # rows_overall = []
    # rows_smf_distrib = []
    # rows_overall_gi = []
    # rows_gi = []
    # for sid, cfg in enumerate(cfgs):
    #     G = nx.read_gpickle(cfg['gpath'])
    #     print("Number of genes in network: %d" % len(G.nodes()))

    #     df = pd.read_csv(cfg['smf_path'])
    #     bins = sorted(set(df['bin'].astype(int)))
    #     breakdown = np.array([np.sum(df['bin'] == b) for b in bins])

    #     sgo = np.load(cfg['sgo_path'])
    #     F = sgo['F']
    #     ix_no_terms = np.sum(F, axis=1) == 0
        
    #     rows_overall.append({
    #         "name" : cfg['name'],
    #         "n_genes" : len(G.nodes()),
    #         "n_no_sgo" : np.sum(ix_no_terms),
    #         "p_no_sgo" : np.mean(ix_no_terms) * 100,
    #         "label_outside" : cfg.get('label_outside', False),
    #         "fsize" : cfg.get('fsize', plot_cfg['bar_label_size'])
    #     })

    #     ix_with_smf = np.zeros(F.shape[0]).astype(bool)
    #     ix_with_smf[df['id']] = True
    #     ix = ix_no_terms & ix_with_smf
    #     n_no_sgo_smf = np.sum(ix)

    #     for b in bins:
    #         ix_in_bin = np.zeros(F.shape[0]).astype(bool)
    #         sdf = df[df['bin'] == b]
    #         ix_in_bin[sdf['id']] = True
    #         ix = ix_no_terms & ix_in_bin

    #         rows_smf_distrib.append({
    #             "name" : cfg['name'],
    #             "bin" : SMF_BINS[b],
    #             "p_no_sgo" : np.sum(ix) * 100 / n_no_sgo_smf,
    #             "n_no_sgo" : np.sum(ix),
    #             "n_with_sgo" : np.sum(~ix_no_terms & ix_in_bin),
    #             "p_tot" : np.sum(ix_in_bin) * 100 / np.sum(ix_with_smf),
    #             "b" : b,
    #             "sid" : sid
    #         })

    #     gi_df = pd.read_csv(cfg['gi_path'])
    #     gi_df['bin'] = (gi_df['bin'] == 1).astype(int)
    #     gi_bins = sorted(set(gi_df['bin']), reverse=True)
    #     breakdown = np.array([np.sum(gi_df['bin'] == b) for b in gi_bins])
        
    #     ix_a_no_sgo = np.sum(F[gi_df['a_id'],:], axis=1) == 0
    #     ix_b_no_sgo = np.sum(F[gi_df['b_id'],:], axis=1) == 0

    #     ix_no_sgo_either = ix_a_no_sgo | ix_b_no_sgo
    #     n_no_sgo_either = np.sum(ix_no_sgo_either)

        
    #     rows_overall_gi.append({
    #         "name" : cfg['name'],
    #         "n_pairs" : gi_df.shape[0],
    #         "n_no_sgo" : n_no_sgo_either,
    #         "p_no_sgo" : np.mean(ix_no_sgo_either) * 100,
    #         "label_outside" : cfg.get('gi_label_outside', False),
    #         "fsize" : cfg.get('gi_fsize', plot_cfg['bar_label_size'])
    #     })


    #     for b in gi_bins:
    #         ix_b = gi_df['bin'] == b
    #         ix = ix_b & ix_no_sgo_either
    #         rows_gi.append({
    #             "name" : cfg['name'],
    #             "bin" : GI_BINS[b],
    #             "p_no_sgo" : np.sum(ix) / n_no_sgo_either,
    #             "n_no_sgo" : np.sum(ix),
    #             "n_with_sgo" : np.sum(ix_b & ~ix_no_sgo_either),
    #             "p_tot" : np.sum(ix_b) / gi_df.shape[0],
    #             "b" : b,
    #             "sid" : sid
    #         })

    # df_overall = pd.DataFrame(rows_overall)
    # df_smf_distrib = pd.DataFrame(rows_smf_distrib)
    # df_overall_gi = pd.DataFrame(rows_overall_gi)
    # df_gi_distrib = pd.DataFrame(rows_gi)

    # df_overall.to_csv('../tmp/df_overall', index=False)
    # df_smf_distrib.to_csv('../tmp/df_smf_distrib', index=False)
    # df_overall_gi.to_csv('../tmp/df_overall_gi', index=False)
    # df_gi_distrib.to_csv('../tmp/df_gi_distrib', index=False)
    

    df_overall = pd.read_csv('../tmp/df_overall')
    df_smf_distrib = pd.read_csv('../tmp/df_smf_distrib')
    df_overall_gi = pd.read_csv('../tmp/df_overall_gi')
    df_gi_distrib = pd.read_csv('../tmp/df_gi_distrib')

    visualize_df_overall(df_overall, output_path + '/no_sgo_perc.png')
    visualize_df_smf_distrib(df_smf_distrib, output_path)

    
    
    visualize_df_overall(df_overall_gi, output_path + '/no_sgo_perc_gi.png', ylabel='% of Pairs')
    visualize_df_gi_distrib(df_gi_distrib, output_path)
    for s in cfgs:
        print("%s Chi2 Test [GI]" % s['name'])
        chi2test_df_smf_distrib(df_gi_distrib[df_gi_distrib['name'] == s['name']], 2)
    

def visualize_df_overall(df_overall, output_path, ylabel='% of Genes'):
    
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    bar = sns.barplot(x="name", y="p_no_sgo", 
        data=df_overall, ax=ax, palette=[s['color'] for s in cfgs], saturation=1)    
    
    # plot labels
    for i, r in df_overall.iterrows():
        val = r['p_no_sgo'] / 2
        fsize = r['fsize']
        
        if r['label_outside']:
            ax.text(i, val * 2 + 2, r['name'], rotation=90, ha="center", va="bottom", fontsize=fsize, weight="bold")
        else:
            ax.text(i, val, r['name'], rotation=90, 
                ha="center", va="center", fontsize=fsize, weight='bold')

    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel(ylabel, fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 100])
    ax.set_xlabel("")
    ax.set_xticklabels([])
    plt.savefig(output_path, bbox_inches='tight', dpi=100, quality=100)
    
def visualize_df_smf_distrib(df_smf_distrib, output_path):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    bar = sns.barplot(x="bin", y="p_tot", hue="name",
        data=df_smf_distrib, ax=ax, color=(1,1,1,0) , saturation=1, linewidth=0, fill=False, edgecolor='black')  
    for i, b in enumerate(bar.patches):
        x, y = b.get_xy()
        h= b.get_height()
        ax.plot([x+0.015, x+b.get_width()-0.015], [y+h, y+h], linestyle='-', linewidth=7, color='grey', label='Baseline' if i == len(bar.patches)-1 else None)

    bar = sns.barplot(x="bin", y="p_no_sgo", hue="name",
        data=df_smf_distrib, ax=ax, palette=[s['color'] for s in cfgs], saturation=1)    
    
    handles,labels=ax.get_legend_handles_labels()
    handles = [handles[0]] + handles[len(cfgs)+1:]
    labels = [labels[0]] + labels[len(cfgs)+1:]
    
    for i, s in enumerate(cfgs):
        print("%s Chi2 Test [SMF]" % s['name'])
        chi2, p, ddof = chi2test_df_smf_distrib(df_smf_distrib[df_smf_distrib['name'] == s['name']], 3)
        n_stars, level = eval_funcs.compute_stars(p, ALPHA, return_level=True)
        labels[i+1] = "%s ($\chi^2_{%d} = %.1f$, $\\rho < %0.4f$)" % (labels[i+1],ddof, chi2, level)

    ax.legend(handles, labels, fontsize=plot_cfg['legend_size'], loc='upper left',frameon=False)


    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel('% of Genes', fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 100])
    ax.set_xlabel('')
    
    plt.savefig(output_path + '/no_sgo_smf_distrib.png', bbox_inches='tight', dpi=100, quality=100)

def visualize_df_gi_distrib(df_gi_distrib, output_path):
    df_gi_distrib_orig = df_gi_distrib

    df_gi_distrib = df_gi_distrib.copy()

    dummy_rows = []
    for sid, cfg in enumerate(cfgs):
        dummy_rows.append({
                    "name" : cfg['name'],
                    "bin" : "",
                    "p_no_sgo" : 0,
                    "n_no_sgo" : 0,
                    "n_with_sgo" : 0,
                    "p_tot" : 0,
                    "b" : 2,
                    "sid" : sid
                })
    df_gi_distrib = pd.concat([df_gi_distrib, pd.DataFrame(dummy_rows)])

    target_ticks = np.array([0.001, 0.05, 0.5, 0.95, 0.999])
    target_logits = np.log(target_ticks / (1-target_ticks)) + 10

    df_gi_distrib['logit_p_no_sgo'] = np.log(df_gi_distrib['p_no_sgo'] / (1-df_gi_distrib['p_no_sgo'])) + 10
    df_gi_distrib['logit_p_tot'] = np.log(df_gi_distrib['p_tot'] / (1-df_gi_distrib['p_tot'])) + 10
    

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    bar = sns.barplot(x="bin", y="logit_p_tot", hue="name",
        data=df_gi_distrib, ax=ax, color=(1,1,1,0) , saturation=1, linewidth=0, fill=False, edgecolor='black')  
    for i, b in enumerate(bar.patches):
        x, y = b.get_xy()
        h= b.get_height()
        ax.plot([x+0.015, x+b.get_width()-0.015], [y+h, y+h], linestyle='-', linewidth=7, color='grey', label='Baseline' if i == len(bar.patches)-1 else None)

    bar = sns.barplot(x="bin", y="logit_p_no_sgo", hue="name",
        data=df_gi_distrib, ax=ax, palette=[s['color'] for s in cfgs], saturation=1)    
    
    handles,labels=ax.get_legend_handles_labels()
    handles = [handles[0]] + handles[len(cfgs)+1:]
    labels = [labels[0]] + labels[len(cfgs)+1:]

    
    for i, s in enumerate(cfgs):
        print("%s Chi2 Test [SMF]" % s['name'])
        chi2, p, ddof = chi2test_df_smf_distrib(df_gi_distrib_orig[df_gi_distrib_orig['name'] == s['name']], 2)
        n_stars, level = eval_funcs.compute_stars(p, ALPHA, return_level=True)
        labels[i+1] = "%s ($\chi^2_{%d} = %.1f$, $\\rho < %0.4f$)" % (labels[i+1], ddof, chi2, level)


    ax.legend(handles, labels, fontsize=plot_cfg['legend_size']*0.95, loc='upper right', frameon=False)
    

    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_ylabel('% of Pairs', fontsize=plot_cfg['ylabel_size'], fontweight='bold')
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    plt.setp(ax.spines.values(), linewidth=plot_cfg['border_size'], color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlabel('')
    ax.set_yticks(target_logits)
    ax.set_yticklabels(['%3.1f' % (t*100) for t in target_ticks])

    plt.savefig(output_path + '/no_sgo_gi_distrib.png', bbox_inches='tight', dpi=100)

def chi2test_df_smf_distrib(df, n_bins):

    # compute observed frequencies (3 bins x 2 kinds of genes)
    f_obs = np.zeros((n_bins, 2))
    for i, r in df.iterrows():

        # without sgo
        f_obs[ r['b'], 0] += r['n_no_sgo']
        # with sgo
        f_obs[ r['b'], 1] += r['n_with_sgo']

    # compute expected frequencies
    col_marginal = np.sum(f_obs, axis=1, keepdims=True) # (3x1)
    row_marginal = np.sum(f_obs, axis=0, keepdims=True) # (1x3)
    total = np.sum(row_marginal)
    f_exp = np.dot(col_marginal / total, row_marginal)

    print(f_obs)
    print(f_exp)

    chisq, p = stats.chisquare(f_obs, f_exp, axis=None)
    print("Chi2 Statistic: %f, p: %f" % (chisq, p))
    ddof = (f_obs.shape[0]-1) * (f_obs.shape[1]-1)

    return chisq, p, ddof
if __name__ == "__main__":
    output_path = sys.argv[1]
    main(cfgs, output_path)
