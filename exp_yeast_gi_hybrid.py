from genericpath import exists
import numpy as np 
import models.train_and_evaluate
import json 
import copy 
import os 

import figure_auc_roc_curve
import figure_cm 
import figure_dev_test_bacc 

def load_spec(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def main():

    """ evaluate via CV on development portion """
    full_spec = load_spec("cfgs/gi_nn_model.json")
    run_cv_on_spec(full_spec, 'full')

    refined_spec = copy.deepcopy(full_spec)
    refined_spec['single_gene_spec']['selected_feature_sets'] = ['topology', 'sgo', 'smf']
    refined_spec['single_gene_spec']['feature_sets']['topology']['selected_features'] = ['lid']
    refined_spec['double_gene_spec']['feature_sets']['pairwise']['selected_features'] = ['spl']
    run_cv_on_spec(refined_spec, 'refined')

    mn_spec = load_spec("cfgs/gi_mn_model.json")
    run_cv_on_spec(mn_spec, 'mn')

    null_spec = { 'target_col' : 'bin', 'class' : 'null' }
    run_cv_on_spec(null_spec, 'null')

    """ train on development and evaluate on final test """
    run_cv_on_spec(full_spec, 'full', 'dev_test')
    run_cv_on_spec(refined_spec, 'refined', 'dev_test')
    run_cv_on_spec(mn_spec, 'mn', 'dev_test')
    run_cv_on_spec(null_spec, 'null', 'dev_test')

    generate_figures("../results/exp_yeast_gi_hybrid/figures")

def run_cv_on_spec(model_spec, name, mode='cv', n_workers=16):
    if name == 'mn':
        dataset_path =  "../generated-data/dataset_yeast_gi_hybrid_mn.feather"
    else:
        dataset_path =  "../generated-data/dataset_yeast_gi_hybrid.feather"
    models.train_and_evaluate.cv(model_spec, 
                                dataset_path, 
                                "../generated-data/splits/dataset_yeast_gi_hybrid_dev_test.npz",
                                mode,
                                "../results/exp_yeast_gi_hybrid/%s_%s" % (mode, name),
                                n_workers=n_workers,
                                no_train=False,
                                sg_path="../generated-data/dataset_yeast_allppc.feather")


def generate_figures(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "D-Full",
                "color": "#b300ff",
                "cm_color": "#b300ff",
                "name" : "full"
            },
            {
                "title": "D-Refined",
                "color": "#FF0000",
                "cm_color": "#FF0000",
                "name" : "refined"
            },
            {
                "title": "D-MN",
                "color": "#3A90FF",
                "name" : "mn"
            },
            {
                "title": "Null",
                "color": "#c9c9c9",
                "star_color": "grey",
                "cm_color": "grey",
                "name" : "null"
            }
        ],
        "classes": [
            "Negative",
            "Neutral",
            "Positive",
            "Suppression"
        ],
        "short_classes": [
            "-",
            "N",
            "+",
            "S"
        ],
        "ylim": [
            0,
            0.75
        ],
        "aspect" : 1
    }
    figure_dev_test_bacc.generate_figures(spec, "../results/exp_yeast_gi_hybrid", "../results/exp_yeast_gi_hybrid/figures/overall_bacc.png")

    for model in spec['models']:
        model['results_path'] = "../results/exp_yeast_gi_hybrid/dev_test_%s/results.json" % model['name']
        figure_cm.plot_cm(model['results_path'], model['color'], spec['short_classes'], 
            "../results/exp_yeast_gi_hybrid/figures/cm_%s.png" % model['name'])
    
    for i in range(len(spec['classes'])):
        figure_auc_roc_curve.plot_auc_roc_curves(spec, i, '../results/exp_yeast_gi_hybrid/figures/curve_auc_roc_dev_test_%s.png' % spec["short_classes"][i])
    
    

if __name__ == "__main__":
    main()
