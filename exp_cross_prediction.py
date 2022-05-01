from venv import create
import models.train_and_evaluate
import json 
import copy 
import os 

import figure_cross_prediction  
import pandas as pd 

def load_spec(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():

    mn_spec = load_spec("cfgs/gi_mn_model.json")
    mn_spec['target_col'] = 'is_neutral'
    run_cv_on_spec(mn_spec, "mn")
    
    evaluate_cv_on_spec(mn_spec, "mn", "rel_not_ppc")
    evaluate_cv_on_spec(mn_spec, "mn", "rel_not_phospho")
    evaluate_cv_on_spec(mn_spec, "mn", "rel_not_trans")
    
    strict_spec = create_strict_spec()
    run_cv_on_spec(strict_spec, "mn_strict")
    evaluate_cv_on_spec(strict_spec, "mn_strict", "rel_not_ppc")
    evaluate_cv_on_spec(strict_spec, "mn_strict", "rel_not_phospho")
    evaluate_cv_on_spec(strict_spec, "mn_strict", "rel_not_trans")

    generate_figures()

def run_cv_on_spec(model_spec, name):
    sg_path = "../generated-data/dataset_yeast_allppc.feather"
    
    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/dataset_yeast_gi_hybrid_mn.feather",
                                "../generated-data/splits/dataset_yeast_gi_hybrid.npz",
                                "cv",
                                "../results/exp_cross_prediction/%s" % name,
                                n_workers=16,
                                no_train=False,
                                sg_path=sg_path)

def evaluate_cv_on_spec(model_spec, name, target_col):
    sg_path = "../generated-data/dataset_yeast_allppc.feather"
    model_spec['target_col'] = target_col
    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/dataset_yeast_gi_hybrid_mn.feather",
                                "../generated-data/splits/dataset_yeast_gi_hybrid.npz",
                                "cv",
                                "../results/exp_cross_prediction/%s" % name,
                                n_workers=16,
                                no_train=True,
                                sg_path=sg_path,
                                results_path="../results/exp_cross_prediction/%s_%s.json" % (name, target_col))

def create_strict_spec():
    mn_spec = load_spec("cfgs/gi_mn_model.json")
    mn_spec['target_col'] = 'is_neutral'

    exclude = ['GO:0016791', 'GO:0016301', 'GO:0008134']

    df = pd.read_feather("../generated-data/dataset_yeast_gi_hybrid_mn.feather")
    sgo_cols = df.columns[df.columns.str.startswith('sgo-')]

    included_cols = [c for c in sgo_cols if c.replace('sgo-','') not in exclude]

    mn_spec['features'] = included_cols + ['smf-']

    return mn_spec

def generate_figures():
    output_dir = "../results/exp_cross_prediction/figures/"
    os.makedirs(output_dir, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "GI",
                "color": "#1f77b4",
                "cm_color": "#1f77b4",
                "results_paths" : [
                    "../results/exp_cross_prediction/mn/results.json",
                    "../results/exp_cross_prediction/mn_strict/results.json",
                ],
                "name" : "gi"
            },
            {
                "title": "Coprecipitation",
                "color": "#ff7f0e",
                "cm_color": "#ff7f0e",
                "results_paths" : [
                    "../results/exp_cross_prediction/mn_rel_not_ppc.json",
                    "../results/exp_cross_prediction/mn_strict_rel_not_ppc.json"
                ],
                "fsize" : 50,
                "name" : "ppc"
            },
            {
                "title": "Phopshorylation",
                "color": "#2ca02c",
                "results_paths" : [
                    "../results/exp_cross_prediction/mn_rel_not_phospho.json",
                    "../results/exp_cross_prediction/mn_strict_rel_not_phospho.json"
                ],
                "name" : "phospho"
            },
            {
                "title": "Transcription",
                "color": "#d62728",
                "results_paths" : [
                    "../results/exp_cross_prediction/mn_rel_not_trans.json",
                    "../results/exp_cross_prediction/mn_strict_rel_not_trans.json",
                ],
                "fsize" : 50,
                "name" : "trans"
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
        "ylim" : [0,0.8],
        "aspect" : 1
    }


    figure_cross_prediction.generate_figures(spec, "../results/exp_cross_prediction", "../results/exp_cross_prediction/figures/overall_bacc.png")


if __name__ == "__main__":
    main()
