import utils.eval_funcs
import sklearn.metrics
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors 

plot_cfg = {
    "tick_label_size" : 50,
    "xlabel_size" : 42,
    "ylabel_size" : 42,
    "annot_size" : 42,
    "title_size" : 38
}


plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["font.weight"] = "bold"
def average_results(cv_dir, thresholds):

    files = utils.eval_funcs.get_files(cv_dir)

    cms_by_t = { t: [] for t in thresholds }
    for file in files:
        print(file)
        d = np.load(file, allow_pickle=True)

        r = d['r'].item()
        y_target = d['y_target']

        for t in thresholds:
            y_pred = d['preds'][:,1] > t
            cm = sklearn.metrics.confusion_matrix(y_target, y_pred)
            cm = cm / np.sum(cm, axis=1, keepdims=True)
            cms_by_t[t].append(cm)

    cms_by_t = np.array([np.mean(cms_by_t[t], axis=0) for t in thresholds])
    return cms_by_t

def sweep_thresholds(cv_dir, thresholds, output_file):

    r = average_results(cv_dir, 1-thresholds)
    np.savez(output_file, thresholds=thresholds, r=r)
    
def visualize_swept_thresholds(file_path, color, output_path, classes=['I', 'N']):

    d = np.load(file_path + '.npz')
    r = d['r']
    thresholds = d['thresholds']

    f, axes = plt.subplots(1, 10, figsize=(40, 10))
    axes = axes.flatten()

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
            ["white",color])
    n_classes = len(classes)
    
    for k, ax in enumerate(axes):

        cm = r[k, :, :]
        t = thresholds[k]

        ax.imshow(cm, cmap=cmap)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, "%0.2f" % cm[i, j], ha="center", va="center", 
                    fontsize=plot_cfg['annot_size'], fontweight='bold')

        xlabels = ax.get_xticks()
        ix = np.isin(xlabels, np.arange(n_classes))
        xlabels = xlabels.astype(str)
        xlabels[ix] = classes
        xlabels[~ix] = ''

        ax.set_xticklabels(xlabels, fontweight='bold', color=color, fontsize=plot_cfg['annot_size'])

        xlabels = ax.get_yticks()

        if k % 10 == 0:
            
            ix = np.isin(xlabels, np.arange(n_classes))
            xlabels = xlabels.astype(str)
            xlabels[ix] = classes
            xlabels[~ix] = ''
        else:
            xlabels = ['' for l in xlabels]

        ax.set_yticklabels(xlabels, fontweight='bold', color=color, fontsize=plot_cfg['annot_size'])
            
        ax.xaxis.set_tick_params(length=0, width=0, which='both')
        ax.yaxis.set_tick_params(length=0, width=0, which='both')
        ax.xaxis.tick_top()
        ax.set_title('At %0.2f' % t, y=-0.20, color='red', fontweight='bold', fontsize=plot_cfg['title_size'])
        plt.setp(ax.spines.values(), linewidth=0)

    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()

    #plt.show()

thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

#
# OR
#
# sweep_thresholds('../results/task_yeast_smf_30_binary/orm', thresholds,
#   '../results/task_yeast_smf_30_binary/swept_thresholds_or')
# sweep_thresholds('../results/task_pombe_smf_binary/orm', thresholds,
#    '../results/task_pombe_smf_binary/swept_thresholds_or')
# sweep_thresholds('../results/task_human_smf_binary/orm', thresholds,
#    '../results/task_human_smf_binary/swept_thresholds_or')
# sweep_thresholds('../results/task_dro_smf_binary/orm', thresholds,
#    '../results/task_dro_smf_binary/swept_thresholds_or')
# sweep_thresholds('../results/task_human_smf_org/orm', thresholds,
#    '../results/task_human_smf_org/swept_thresholds_or')
# sweep_thresholds('../results/task_dro_smf_org/orm', thresholds,
#    '../results/task_dro_smf_org/swept_thresholds_or')

visualize_swept_thresholds('../results/task_yeast_smf_30_binary/swept_thresholds_or', "#00CC00", "../figures/yeast_smf_or.png", classes=['L', 'V'])
visualize_swept_thresholds('../results/task_pombe_smf_binary/swept_thresholds_or', "#00CC00", "../figures/pombe_or.png", classes=['L', 'V'])
visualize_swept_thresholds('../results/task_human_smf_binary/swept_thresholds_or', "#00CC00", "../figures/human_smf_or.png", classes=['L', 'V'])
visualize_swept_thresholds('../results/task_dro_smf_binary/swept_thresholds_or', "#00CC00", "../figures/dro_smf_or.png", classes=['L', 'V'])
visualize_swept_thresholds('../results/task_human_smf_org/swept_thresholds_or', "#00CC00", "../figures/human_smf_or_org.png", classes=['L', 'V'])
visualize_swept_thresholds('../results/task_dro_smf_org/swept_thresholds_or', "#00CC00", "../figures/dro_smf_or_org.png", classes=['L', 'V'])



# sweep_thresholds('../results/task_yeast_gi_hybrid_binary/mn', thresholds,
#    '../results/task_yeast_gi_hybrid_binary/swept_thresholds_mn')
# sweep_thresholds('../results/task_yeast_gi_costanzo_binary/mn', thresholds,
#    '../results/task_yeast_gi_costanzo_binary/swept_thresholds_mn')
# sweep_thresholds('../results/task_pombe_gi_binary/mn', thresholds,
#    '../results/task_pombe_gi_binary/swept_thresholds_mn')
# sweep_thresholds('../results/task_human_gi/mn', thresholds,
#    '../results/task_human_gi/swept_thresholds_mn')
# sweep_thresholds('../results/task_dro_gi/mn', thresholds,
#    '../results/task_dro_gi/swept_thresholds_mn')


# visualize_swept_thresholds('../results/task_yeast_gi_hybrid_binary/swept_thresholds_mn', "#3A90FF", "../figures/hybrid_mn_thres.png")
# visualize_swept_thresholds('../results/task_yeast_gi_costanzo_binary/swept_thresholds_mn', "#3A90FF", "../figures/costanzo_mn_thres.png")
# visualize_swept_thresholds('../results/task_pombe_gi_binary/swept_thresholds_mn', "#3A90FF", "../figures/pombe_mn_thres.png")
# visualize_swept_thresholds('../results/task_human_gi/swept_thresholds_mn', "#3A90FF", "../figures/human_mn_thres.png")
# visualize_swept_thresholds('../results/task_dro_gi/swept_thresholds_mn', "#3A90FF", "../figures/dro_mn_thres.png")
