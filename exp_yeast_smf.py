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
    full_spec = load_spec("cfgs/smf_nn_model.json")
    run_cv_on_spec(full_spec, 'full')

    refined_spec = copy.deepcopy(full_spec)
    refined_spec['selected_feature_sets'] = ['topology', 'sgo', 'redundancy']
    refined_spec['feature_sets']['topology']['selected_features'] = ['lid']
    refined_spec['feature_sets']['redundancy']['selected_features'] = ['pident']
    run_cv_on_spec(refined_spec, 'refined')

    mn_spec = load_spec("cfgs/smf_mn_model.json")
    run_cv_on_spec(mn_spec, 'mn')

    null_spec = { 'target_col' : 'bin', 'class' : 'null' }
    run_cv_on_spec(null_spec, 'null')
    

    """ train on development and evaluate on final test """
    run_cv_on_spec(full_spec, 'full', 'dev_test')
    run_cv_on_spec(refined_spec, 'refined', 'dev_test')
    run_cv_on_spec(mn_spec, 'mn', 'dev_test')
    run_cv_on_spec(null_spec, 'null', 'dev_test')

    """ generate figures """
    generate_figures("../results/exp_yeast_smf/figures")

def run_cv_on_spec(model_spec, name, mode='cv'):
    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/dataset_yeast_smf.feather", 
                                "../generated-data/splits/dataset_yeast_smf_dev_test.npz",
                                mode,
                                "../results/exp_yeast_smf/%s_%s" % (mode, name),
                                n_workers=32,
                                no_train=False)


def generate_figures(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "S-Full",
                "color": "#b300ff",
                "cm_color": "#b300ff",
                "name" : "full"
            },
            {
                "title": "S-Refined",
                "color": "#FF0000",
                "cm_color": "#FF0000",
                "name" : "refined"
            },
            {
                "title": "S-MN",
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
            "Lethal",
            "Reduced growth",
            "Normal"
        ],
        "short_classes": [
            "L",
            "R",
            "N"
        ],
        "ylim": [
            0,
            0.75
        ],
        "aspect" : 1
    }
    figure_dev_test_bacc.generate_figures(spec, "../results/exp_yeast_smf", "../results/exp_yeast_smf/figures/overall_bacc.png")

    for model in spec['models']:
        model['results_path'] = "../results/exp_yeast_smf/dev_test_%s/results.json" % model['name']
        figure_cm.plot_cm(model['results_path'], model['color'], spec['short_classes'], 
            "../results/exp_yeast_smf/figures/cm_%s.png" % model['name'])
    
    for i in range(len(spec['classes'])):
        figure_auc_roc_curve.plot_auc_roc_curves(spec, i, '../results/exp_yeast_smf/figures/curve_auc_roc_dev_test_%s.png' % spec["short_classes"][i])
    
    
# def summarize_results(model_output_path, class_labels):
#     with open("%s/results.json" % model_output_path, 'r') as f:
#         r = json.load(f)

#         results = r['results']

#         row = {}

#         row['bacc'] = np.mean([e['bacc'] for e in results])
#         row['acc'] = np.mean([e['acc'] for e in results])
#         row['cm'] = np.mean([e['cm'] for e in results], axis=0)

#         num_classes = len(class_labels)
            
#         for c in range(num_classes):
#             row['%s_auc_roc' % class_labels[c]] = np.mean([e['auc_roc'][c] for e in results])
#             row['%s_bacc' % class_labels[c]] = np.mean([e['per_class_bacc'][c] for e in results])
#             row['%s_auc_pr' % class_labels[c]] = np.mean([e['pr'][c] for e in results])
                
#     print(row)

if __name__ == "__main__":
    main()
