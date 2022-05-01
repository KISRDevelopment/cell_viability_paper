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


    full_spec = load_spec("cfgs/gi_nn_model.json")
    # run_cv_on_spec(full_spec, 'full', 'yeast')

    refined_spec = copy.deepcopy(full_spec)
    refined_spec['single_gene_spec']['selected_feature_sets'] = ['topology', 'sgo', 'smf']
    refined_spec['single_gene_spec']['feature_sets']['topology']['selected_features'] = ['lid']
    refined_spec['double_gene_spec']['feature_sets']['pairwise']['selected_features'] = ['spl']
    # run_cv_on_spec(refined_spec, 'refined', 'yeast')
    # run_cv_on_spec(refined_spec, 'refined', 'pombe')

    mn_spec = load_spec("cfgs/gi_mn_model.json")
    # run_cv_on_spec(mn_spec, 'mn', 'yeast')
    # run_cv_on_spec(mn_spec, 'mn', 'pombe')
    
    null_spec = { 'target_col' : 'bin', 'class' : 'null' }
    # run_cv_on_spec(null_spec, 'null', 'yeast')
    # run_cv_on_spec(null_spec, 'null', 'pombe')

    generate_figures('yeast')
    generate_figures('pombe')

def run_cv_on_spec(model_spec, name, org, n_workers=16):
    
    postfix = ''
    if name == 'mn':
        postfix = '_mn'

    dataset = "%s_gi" % org
    sg_path = "%s_smf" % org 
    if org == 'yeast':
        dataset = "yeast_gi_costanzo"
        sg_path = "yeast_allppc"
    
    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/dataset_%s%s.feather" % (dataset, postfix), 
                                "../generated-data/splits/dataset_%s.npz" % dataset,
                                "cv",
                                "../results/exp_gi_costanzo_pombe/%s_%s" % (name, org),
                                n_workers=n_workers,
                                no_train=False,
                                sg_path="../generated-data/dataset_%s.feather" % sg_path)


def generate_figures(org):

    os.makedirs("../results/exp_gi_costanzo_pombe/figures/%s" % org, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "D-Full",
                "color": "#b300ff",
                "cm_color": "#b300ff",
                "name" : "full_%s" % org 
            },
            {
                "title": "D-Refined",
                "color": "#FF0000",
                "cm_color": "#FF0000",
                "name" : "refined_%s" % org
            },
            {
                "title": "D-MN",
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
            0.6
        ],
        "aspect" : 1
    }

    if org == 'pombe':
        spec['models'] = spec['models'][1:]
    
    figure_cv_bacc.generate_figures(spec, "../results/exp_gi_costanzo_pombe", 
        "../results/exp_gi_costanzo_pombe/figures/%s/overall_bacc.png" % org)

    for model in spec['models']:
        model['results_path'] = "../results/exp_gi_costanzo_pombe/%s/results.json" % model['name']
        figure_cm.plot_cm(model['results_path'], model['color'], spec['short_classes'], 
            "../results/exp_gi_costanzo_pombe/figures/%s/cm_%s.png" % (org, model['name']))
    
    for i in range(len(spec['classes'])):
        figure_auc_roc_curve.plot_auc_roc_curves(spec, i, 
            '../results/exp_gi_costanzo_pombe/figures/%s/curve_auc_roc_%s.png' % (org, spec["short_classes"][i]))
    
    

if __name__ == "__main__":
    main()
