from genericpath import exists
import numpy as np 
import models.train_and_evaluate
import json 
import copy 
import os 

import figure_auc_roc_curve
import figure_cm 
import figure_cv_bacc 

def load_spec(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():

    full_spec = load_spec("cfgs/smf_nn_model.json")
    refined_spec = copy.deepcopy(full_spec)
    refined_spec['selected_feature_sets'] = ['topology', 'sgo', 'redundancy']
    refined_spec['feature_sets']['topology']['selected_features'] = ['lid']
    refined_spec['feature_sets']['redundancy']['selected_features'] = ['pident']
    # for org in ['pombe', 'human', 'dro']:
    #    run_cv_on_spec(refined_spec, 'refined', org)

    mn_spec = load_spec("cfgs/smf_mn_model.json")
    # for org in ['pombe', 'human', 'dro']:
    #     run_cv_on_spec(mn_spec, 'mn', org)
    
    null_spec = { 'target_col' : 'bin', 'class' : 'null' }
    # for org in ['pombe', 'human', 'dro']:
    #     run_cv_on_spec(null_spec, 'null', org)
        
    for org in ['pombe', 'human', 'dro']:
        generate_figures(org)

def run_cv_on_spec(model_spec, name, org, mo_v=False):
    postfix = '_mo_v' if mo_v else ''

    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/dataset_%s_smf%s.feather" % (org, postfix), 
                                "../generated-data/splits/dataset_%s_smf%s.npz" % (org, postfix),
                                "cv",
                                "../results/exp_smf_other_orgs/%s_%s%s" % (name, org, postfix),
                                n_workers=25,
                                no_train=False)

def generate_figures(org):
    output_dir = "../results/exp_smf_other_orgs/figures/%s" % org
    os.makedirs(output_dir, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "S-Refined",
                "color": "#FF0000",
                "cm_color": "#FF0000",
                "name" : "refined_%s" % org
            },
            {
                "title": "S-MN",
                "color": "#3A90FF",
                "name" : "mn_%s" % org
            },
            {
                "title": "Null",
                "color": "#c9c9c9",
                "star_color": "grey",
                "cm_color": "grey",
                "name" : "null_%s" % org
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
            "V"
        ],
        "ylim" : [0,0.65],
        "aspect" : 1
    }

    figure_cv_bacc.generate_figures(spec, "../results/exp_smf_other_orgs", os.path.join(output_dir, 'overall_bacc.png'))

    for model in spec['models']:
        model['results_path'] = "../results/exp_smf_other_orgs/%s/results.json" % (model['name'])
        figure_cm.plot_cm(model['results_path'], model['color'], spec['short_classes'], os.path.join(output_dir, "cm_%s.png" % model['name']))
    
    for i in range(len(spec['classes'])):
        figure_auc_roc_curve.plot_auc_roc_curves(spec, i, os.path.join(output_dir, "auc_roc%s.png" % spec["short_classes"][i]))
    
    


if __name__ == "__main__":
    main()
