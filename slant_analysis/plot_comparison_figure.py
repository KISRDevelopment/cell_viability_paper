import numpy as np 
import pandas as pd 
import utils.eval_funcs as eval_funcs
import pprint 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors 
from scipy import interp
import sklearn.metrics 

plot_cfg = {
    "tick_label_size" : 50,
    "xlabel_size" : 60,
    "ylabel_size" : 60,
    "border_size" : 6,
    "bar_border_size" : 2.5,
    "bar_label_size" : 38,
    "stars_label_size" : 48,
    "annot_size" : 72,
    "max_cm_classes" : 4,
    "max_bars" : 4,
    "legend_size" : 38
}

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"

def main():

    slant_results, slant_cm = evaluate_slant("/home/mmkhajah/Downloads/slant_data_dir/data_dir/training/yeast_testing_sl_all.csv", "/home/mmkhajah/Downloads/slant_data_dir/data_dir/results/recent/yeast_predictions_sl_all.csv")
    gnn_results, gnn_cm = evaluate_slant("/home/mmkhajah/Downloads/slant_data_dir/data_dir/training/yeast_testing_sl_all.csv", "../results/slant_generic_nn.csv")
    
    cv_slant_results = evaluate_ours("../results/task_yeast_gi_hybrid_binary/slant_final")
    cv_refined_results = evaluate_ours("../results/task_yeast_gi_hybrid_binary/refined")
    cv_mn_results = evaluate_ours("../results/task_yeast_gi_hybrid_binary/mn")

    slant_results['name'] = 'SLant Data \n SLant Model'
    slant_results['color'] = '#c5ebfe'
    slant_results['classes'] = ['N', '-']
    slant_results['pos_class'] = 1

    gnn_results['name'] = 'SLant Data \n Neural Network'
    gnn_results['color'] = 'cyan'
    gnn_results['classes'] = ['N', '-']
    gnn_results['pos_class'] = 1
    
    cv_slant_results['name'] = 'Our Data \n SLant Features'
    cv_slant_results['color'] = 'orange'
    cv_slant_results['classes'] = ['I', 'N']
    cv_slant_results['pos_class'] = 0
    
    cv_refined_results['name'] = 'Refined Model'
    cv_refined_results['color'] = '#FF0000'
    cv_refined_results['as_line'] = True 
    cv_refined_results['classes'] = ['I', 'N']

    cv_mn_results['name'] = 'MN'
    cv_mn_results['color'] = '#3A90FF'
    cv_mn_results['as_line'] = True 
    cv_mn_results['classes'] = ['I', 'N']

    #plot_overall_bacc([slant_results, gnn_results, cv_slant_results, cv_refined_results, cv_mn_results], '../tmp/slant_overall_bacc.png')
    #plot_cms([slant_results, gnn_results, cv_slant_results], '../tmp/slant_cm_')
    plot_roc([slant_results, gnn_results, cv_slant_results], '../tmp/slant_roc.png')
def plot_overall_bacc(results, output_path):

    colors = [r['color'] for r in results]
    rows = []
    for r in results:
        if not r.get('as_line', False):
            rows.append({
                "model" : r['name'],
                "bacc" : r['bacc'],
            })
    
    rem_bars = 5 - len(rows) 
    for i in range(rem_bars):
        rows.append({"model" : "%d" % i, "bacc" : 0  })
    
    df = pd.DataFrame(rows)
    
    g = sns.catplot(x="model", y="bacc", data=df,
        kind="bar",
        height=10,
        aspect=1,
        palette=colors,
        edgecolor='black',
        errwidth=5,
        errcolor='black',
        linewidth=plot_cfg["bar_border_size"],
        saturation=1)
    
    ax = g.ax

    for r in results:
        if r.get('as_line', False):
            ax.plot([-0.5, len(rows)+0.5], [r['bacc'], r['bacc']], '--', color=r['color'], linewidth=5)

    for i, m in enumerate(results):
        a_ix = df['model'] == m['name']
        bacc = (np.mean(df[a_ix]['bacc']) - np.std(df[a_ix]['bacc'])) / 2
        ax.text(i, bacc, m['name'], rotation=90, ha="center", va="center", fontsize=plot_cfg['bar_label_size'], weight='bold')

    # legend_handles = []
    # for r in results:
    #     patch = mpatches.Patch(color=r['color'], label=r['name'])
    #     legend_handles.append(patch)
    # ax.legend(handles=legend_handles, ncol=1, 
    #     bbox_to_anchor=(0.5, -0.02),
    #     loc='upper center', 
    #     fontsize=plot_cfg['legend_size'], frameon=False)


    ax.yaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"])
    ax.xaxis.set_tick_params(labelsize=plot_cfg["tick_label_size"])
    ax.set_xlabel('')
    ax.set_ylabel('Balanced Accuracy', fontsize=plot_cfg["ylabel_size"], weight='bold')
    ax.set_xticklabels([])

    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    ax.set_ylim([0.0, 1.0])
    plt.setp(ax.spines.values(), linewidth=plot_cfg["border_size"], color='black')

    plt.savefig(output_path, bbox_inches='tight', dpi=100)

    #plt.show()

def plot_cms(results, output_path):

    for r in results:
        cm = r['cm']
        cm = cm / np.sum(cm, axis=1, keepdims=True)
        classes = r['classes']

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
            ["white",r['color']])

        f, ax = plt.subplots(figsize=(10, 10))

        displayed_cm = np.zeros((plot_cfg['max_cm_classes'], plot_cfg['max_cm_classes']))
        displayed_cm[:cm.shape[0], :cm.shape[0]] = cm 

        ax.imshow(displayed_cm, cmap=cmap)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, "%0.2f" % cm[i, j], ha="center", va="center", 
                fontsize=plot_cfg['annot_size'])

        xlabels = ax.get_xticks()
        ix = np.isin(xlabels, np.arange(len(classes)))
        xlabels = xlabels.astype(str)
        xlabels[ix] = classes
        xlabels[~ix] = ''

        ax.set_xticklabels(xlabels, fontsize=plot_cfg['annot_size'])
        ax.set_yticklabels(xlabels, fontsize=plot_cfg['annot_size'])
        
        ax.xaxis.set_tick_params(length=0, width=0, which='both', colors=r['color'])
        ax.yaxis.set_tick_params(length=0, width=0, which='both', colors=r['color'])
        ax.xaxis.tick_top()
        plt.setp(ax.spines.values(), linewidth=0)

        plt.savefig("%s_%s.png" % (output_path, r['name']), bbox_inches='tight')
        #
        # plt.show() 


def plot_roc(results, output_path):

    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    for r in results:
        fpr, roc_curve = r['fpr'], r['roc_curve']
        ax.plot(fpr, roc_curve, linewidth=7, color=r['color'])

    ax.plot(fpr, fpr, linewidth=2, color='black', linestyle='dashed')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.yaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.xaxis.set_tick_params(labelsize=plot_cfg['tick_label_size'])
    ax.set_xlabel('False Positive Rate', fontsize=plot_cfg['xlabel_size'], weight='heavy')
    ax.set_ylabel('True Positive Rate', fontsize=plot_cfg['ylabel_size'], weight='heavy')
    ax.yaxis.set_tick_params(length=10, width=1, which='both')
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    plt.setp(ax.spines.values(), linewidth=6, color='black')

    plt.savefig("%s.png" % (output_path), bbox_inches='tight', dpi=100)


def evaluate_ours(cv_path, klass=0):

    r = eval_funcs.average_results(cv_path)
    fpr, roc_curve = eval_funcs.average_roc_curve(cv_path, klass)

    r['fpr'] = fpr 
    r['roc_curve'] = roc_curve

    return r 

def evaluate_slant(dataset_path, preds_path):
    

    test_set_df = pd.read_csv(dataset_path, sep='\t')
    result_df = pd.read_csv(preds_path, sep=' ')
   
    assert test_set_df.shape[0] == result_df.shape[0]

    sl_preds = np.array(result_df[['X0', 'X1']]).astype(float)
    y_true = np.array(test_set_df['sl'] == 'X1').astype(int)

    hardpreds = np.argmax(sl_preds, axis=1)

    eval_results, cm = eval_funcs.eval_classifier(y_true, sl_preds)

    BASE_FPR = np.linspace(0, 1, 101)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, sl_preds[:,1])
    tpr = interp(BASE_FPR, fpr, tpr)
    tpr[0] = 0.0

    eval_results['fpr'] = BASE_FPR 
    eval_results['roc_curve'] = tpr

    return eval_results, cm 

if __name__ == "__main__":
    main()
