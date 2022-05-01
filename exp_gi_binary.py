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
    refined_spec = copy.deepcopy(full_spec)
    refined_spec['single_gene_spec']['selected_feature_sets'] = ['topology', 'sgo', 'smf']
    refined_spec['single_gene_spec']['feature_sets']['topology']['selected_features'] = ['lid']
    refined_spec['double_gene_spec']['feature_sets']['pairwise']['selected_features'] = ['spl']
    refined_spec['target_col'] = 'is_neutral'
    #for org in ['yeast', 'pombe', 'human', 'dro']:
    #   run_cv_on_spec(refined_spec, 'refined', org)

    refined_no_sgo_spec = copy.deepcopy(refined_spec)
    refined_no_sgo_spec['single_gene_spec']['selected_feature_sets'] = ['topology', 'smf']
    # for org in ['yeast', 'pombe', 'human', 'dro']:
    #    run_cv_on_spec(refined_no_sgo_spec, 'refined_no_sgo', org)

    mn_spec = load_spec("cfgs/gi_mn_model.json")
    mn_spec['target_col'] = 'is_neutral'
    # for org in ['yeast', 'pombe', 'human', 'dro']:
    #    run_cv_on_spec(mn_spec, 'mn', org)
    
    mn_spec_no_sgo = copy.deepcopy(mn_spec)
    mn_spec_no_sgo['features'].remove('sgo-')
    # for org in ['yeast', 'pombe', 'human', 'dro']:
    #    run_cv_on_spec(mn_spec_no_sgo, 'mn_no_sgo', org)

    null_spec = { 'target_col' : 'is_neutral', 'class' : 'null' }
    # for org in ['yeast', 'pombe', 'human', 'dro']:
    #    run_cv_on_spec(null_spec, 'null', org)

    for org in ['yeast', 'pombe', 'human', 'dro']:
       generate_figures(org)
    
def run_cv_on_spec(model_spec, name, org):
    
    dataset_name = "%s_gi" % org 
    sg_path = "../generated-data/dataset_%s_smf.feather" % org 
    if org == 'yeast':
        dataset_name = "yeast_gi_hybrid"
        sg_path = "../generated-data/dataset_yeast_allppc.feather"
    
    postfix = ''
    if 'mn' in name:
        postfix = '_mn'

    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/dataset_%s%s.feather" % (dataset_name, postfix), 
                                "../generated-data/splits/dataset_%s.npz" % dataset_name,
                                "cv",
                                "../results/exp_gi_binary/%s_%s" % (name, org),
                                n_workers=16,
                                no_train=False,
                                sg_path=sg_path)

def generate_figures(org):
    output_dir = "../results/exp_gi_binary/figures/%s" % org
    os.makedirs(output_dir, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "D-Refined",
                "color": "#FF0000",
                "cm_color": "#FF0000",
                "name" : "refined_%s" % org
            },
            {
                "title": "D-Refined No sGO",
                "color": "#ffb700",
                "cm_color": "#ffb700",
                "name" : "refined_no_sgo_%s" % org,
                "fsize" : 50
            },
            {
                "title": "D-MN",
                "color": "#3A90FF",
                "name" : "mn_%s" % org
            },
            {
                "title": "D-MN No sGO",
                "color": "#38fffc",
                "name" : "mn_no_sgo_%s" % org,
                "fsize" : 50
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
            "Interacting",
            "Neutral"
        ],
        "short_classes": [
            "I",
            "N"
        ],
        "ylim" : [0,1],
        "aspect" : 1
    }

    figure_cv_bacc.generate_figures(spec, "../results/exp_gi_binary", os.path.join(output_dir, 'overall_bacc.png'))

    for model in spec['models']:
        model['results_path'] = "../results/exp_gi_binary/%s/results.json" % (model['name'])
        figure_cm.plot_cm(model['results_path'], model['color'], spec['short_classes'], os.path.join(output_dir, "cm_%s.png" % model['name']))
    
    for i in range(len(spec['classes'])):
        figure_auc_roc_curve.plot_auc_roc_curves(spec, i, os.path.join(output_dir, "auc_roc%s.png" % spec["short_classes"][i]))
    
    


if __name__ == "__main__":
    main()
