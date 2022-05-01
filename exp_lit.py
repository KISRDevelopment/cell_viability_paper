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

    """ Model Comparisons on the SMF task """
    mn_spec = load_spec("cfgs/smf_mn_model.json")
    mn_spec['target_col'] = 'is_viable'
    run_smf_cv_on_spec(mn_spec, 's-mn')

    campos_spec = load_spec("cfgs/smf_campos_model.json")
    run_smf_cv_on_spec(campos_spec, 'smf_campos')

    lou_spec = load_spec("cfgs/smf_lou_model.json")
    run_smf_cv_on_spec(lou_spec, "smf_lou")

    mistry_spec = load_spec("cfgs/smf_mistry_model.json")
    run_smf_cv_on_spec(mistry_spec, "smf_mistry")

    generate_smf_figures()
    
    """ Model Comparisons on the Negative vs All GI task """
    mn_spec = load_spec("cfgs/gi_mn_model.json")
    mn_spec['target_col'] = 'is_not_negative'
    run_gi_cv_on_spec(mn_spec, "d-mn")

    slant_spec = load_spec("cfgs/gi_slant_model.json")
    run_gi_cv_on_spec(slant_spec, "gi_slant")

    yu_spec = load_spec("cfgs/gi_yu_model.json")
    run_gi_cv_on_spec(yu_spec, "gi_yu", n_workers=10)

    alanis_spec = load_spec("cfgs/gi_alanis-lobato_model.json")
    run_gi_cv_on_spec(alanis_spec, "gi_alanis-lobato")

    generate_gi_figures()

def run_smf_cv_on_spec(model_spec, name):
    postfix = '_lit'
    if name == 's-mn':
        postfix = ''
    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/dataset_yeast_smf%s.feather" % postfix,
                                "../generated-data/splits/dataset_yeast_smf.npz",
                                "cv",
                                "../results/exp_lit/%s" % name,
                                n_workers=16,
                                no_train=False)

def run_gi_cv_on_spec(model_spec, name, n_workers=16):
    postfix = '_lit'
    sg_path = "../generated-data/dataset_yeast_allppc.feather"
    if name == 'd-mn':
        postfix = '_mn'
    if name == 'gi_yu':
        sg_path = "../generated-data/dataset_yeast_smf_yu.feather"
    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/dataset_yeast_gi_hybrid%s.feather" % postfix,
                                "../generated-data/splits/dataset_yeast_gi_hybrid.npz",
                                "cv",
                                "../results/exp_lit/%s" % name,
                                n_workers=n_workers,
                                no_train=False,
                                sg_path=sg_path)

def generate_smf_figures():
    output_dir = "../results/exp_lit/figures/smf"
    os.makedirs(output_dir, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "S-MN",
                "color": "#3A90FF",
                "name" : "s-mn"
            },
            {
                "title": "Campos 2019",
                "color": "#b300ff",
                "name" : "smf_campos",
                "fsize" : 50
            },
            {
                "title": "Mistry 2017",
                "color": "orange",
                "name" : "smf_mistry"
            },
            {
                "title": "Lou 2015",
                "color": "#FF0000",
                "name" : "smf_lou"
            }
        ],
        "classes": [
            "Lethal",
            "Viable"
        ],
        "short_classes": [
            "L",
            "V"
        ],
        "ylim" : [0,1],
        "aspect" : 1
    }

    figure_cv_bacc.generate_figures(spec, "../results/exp_lit", os.path.join(output_dir, 'overall_bacc.png'))

    for model in spec['models']:
        model['results_path'] = "../results/exp_lit/%s/results.json" % (model['name'])
        figure_cm.plot_cm(model['results_path'], model['color'], spec['short_classes'], os.path.join(output_dir, "cm_%s.png" % model['name']))
    
    for i in range(len(spec['classes'])):
        figure_auc_roc_curve.plot_auc_roc_curves(spec, i, os.path.join(output_dir, "auc_roc%s.png" % spec["short_classes"][i]))
    
    
def generate_gi_figures():
    output_dir = "../results/exp_lit/figures/gi"
    os.makedirs(output_dir, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "D-MN",
                "color": "#3A90FF",
                "name" : "d-mn"
            },
            {
                "title": "Benstead-Hume 2019",
                "color": "#b300ff",
                "name" : "gi_slant",
                "fsize" : 40
            },
            {
                "title": "Yu 2015",
                "color": "orange",
                "name" : "gi_yu"
            },
            {
                "title": "Alanis-Lobato 2013",
                "color": "#FF0000",
                "name" : "gi_alanis-lobato",
                "fsize" : 40
            }
        ],
        "classes": [
            "Negative GI",
            "All"
        ],
        "short_classes": [
            "-",
            "All"
        ],
        "ylim" : [0,1],
        "aspect" : 1
    }

    figure_cv_bacc.generate_figures(spec, "../results/exp_lit", os.path.join(output_dir, 'overall_bacc.png'))

    for model in spec['models']:
        model['results_path'] = "../results/exp_lit/%s/results.json" % (model['name'])
        figure_cm.plot_cm(model['results_path'], model['color'], spec['short_classes'], os.path.join(output_dir, "cm_%s.png" % model['name']))
    
    for i in range(len(spec['classes'])):
        figure_auc_roc_curve.plot_auc_roc_curves(spec, i, os.path.join(output_dir, "auc_roc%s.png" % spec["short_classes"][i]))
    
    

if __name__ == "__main__":
    main()
