from venv import create
import models.train_and_evaluate
import json 
import copy 
import os 

import figure_feature_contribution  
import pandas as pd 

def load_spec(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    os.makedirs("../results/exp_mn_feature_contribution/figures", exist_ok=True)

    mn_spec = load_spec("cfgs/smf_mn_model.json")

    run_cv_on_spec(mn_spec, "smf", None, "dataset_yeast_smf", "dataset_yeast_smf")

    run_cv_on_spec(mn_spec, "smf_no_lid", "topology-lid", "dataset_yeast_smf", "dataset_yeast_smf")
    run_cv_on_spec(mn_spec, "smf_no_sgo", "sgo-", "dataset_yeast_smf", "dataset_yeast_smf")
    run_cv_on_spec(mn_spec, "smf_no_redundancy", "redundancy-pident", "dataset_yeast_smf", "dataset_yeast_smf")

    mn_spec = load_spec("cfgs/gi_mn_model.json")

    run_cv_on_spec(mn_spec, "gi", None, "dataset_yeast_gi_hybrid_mn", "dataset_yeast_gi_hybrid")

    run_cv_on_spec(mn_spec, "gi_no_lid", "topology-", "dataset_yeast_gi_hybrid_mn", "dataset_yeast_gi_hybrid")
    run_cv_on_spec(mn_spec, "gi_no_sgo", "sgo-", "dataset_yeast_gi_hybrid_mn", "dataset_yeast_gi_hybrid")
    run_cv_on_spec(mn_spec, "gi_no_smf", "smf-", "dataset_yeast_gi_hybrid_mn", "dataset_yeast_gi_hybrid")
    run_cv_on_spec(mn_spec, "gi_no_spl", "pairwise-spl", "dataset_yeast_gi_hybrid_mn", "dataset_yeast_gi_hybrid")

    mn_spec = load_spec("cfgs/tgi_mn_model.json")

    run_cv_on_spec(mn_spec, "tgi", None, "dataset_yeast_tgi_mn", "dataset_yeast_tgi")

    run_cv_on_spec(mn_spec, "tgi_no_lid", "topology-", "dataset_yeast_tgi_mn", "dataset_yeast_tgi")
    run_cv_on_spec(mn_spec, "tgi_no_sgo", "sgo-", "dataset_yeast_tgi_mn", "dataset_yeast_tgi")
    run_cv_on_spec(mn_spec, "tgi_no_smf", "smf-", "dataset_yeast_tgi_mn", "dataset_yeast_tgi")
    run_cv_on_spec(mn_spec, "tgi_no_spl", "abc-pairwise-spl", "dataset_yeast_tgi_mn", "dataset_yeast_tgi")

    generate_smf_figures()
    generate_gi_figures()
    generate_tgi_figures()
def run_cv_on_spec(model_spec, name, feature_to_exclude, dataset_path, split_path):
    sg_path = "../generated-data/dataset_yeast_allppc.feather"
    
    model_spec = copy.copy(model_spec)
    model_spec['features'] = [f for f in model_spec['features'] if f != feature_to_exclude]

    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/%s.feather" % dataset_path,
                                "../generated-data/splits/%s.npz" % split_path,
                                "cv",
                                "../results/exp_mn_feature_contribution/%s" % name,
                                n_workers=16,
                                no_train=False,
                                sg_path=sg_path)

def create_strict_spec():
    mn_spec = load_spec("cfgs/gi_mn_model.json")
    mn_spec['target_col'] = 'is_neutral'

    exclude = ['GO:0016791', 'GO:0016301', 'GO:0008134']

    df = pd.read_feather("../generated-data/dataset_yeast_gi_hybrid_mn.feather")
    sgo_cols = df.columns[df.columns.str.startswith('sgo-')]

    included_cols = [c for c in sgo_cols if c.replace('sgo-','') not in exclude]

    mn_spec['features'] = included_cols + ['smf-']

    return mn_spec

def generate_smf_figures():
    spec = {
        "models": [
            {
                "title": "S-MN",
                "color": "#FF0000",
                "name" : "smf"
            },
            {
                "title": "No LID",
                "color": "magenta",
                "name" : "smf_no_lid"
            },
            {
                "title": "No sGO",
                "color": "cyan",
                "name" : "smf_no_sgo"
            },
            {
                "title": "No Homology",
                "color": "#d62728",
                "name" : "smf_no_redundancy"
            },
        ],
        "ylim" : [0,25],
        "aspect" : 1
    }

    figure_feature_contribution.generate_figures(spec, 
        "../results/exp_mn_feature_contribution", 
        "../results/exp_mn_feature_contribution/figures/smf_bacc_drop.png")

def generate_gi_figures():
    spec = {
        "models": [
            {
                "title": "D-MN",
                "color": "#FF0000",
                "name" : "gi"
            },
            {
                "title": "No LID",
                "color": "magenta",
                "name" : "gi_no_lid"
            },
            {
                "title" : "No SPL",
                "color" : "#9467bd",
                "name" : "gi_no_spl"
            },
            {
                "title": "No sGO",
                "color": "cyan",
                "name" : "gi_no_sgo"
            },
            {
                "title": "No SMF",
                "color": "orange",
                "name" : "gi_no_smf"
            },
        ],
        "ylim" : [0,25],
        "aspect" : 1
    }

    figure_feature_contribution.generate_figures(spec, 
        "../results/exp_mn_feature_contribution", 
        "../results/exp_mn_feature_contribution/figures/gi_bacc_drop.png")

def generate_tgi_figures():
    spec = {
        "models": [
            {
                "title": "D-MN",
                "color": "#FF0000",
                "name" : "tgi"
            },
            {
                "title": "No LID",
                "color": "magenta",
                "name" : "tgi_no_lid"
            },
            {
                "title" : "No SCL",
                "color" : "#9467bd",
                "name" : "tgi_no_spl"
            },
            {
                "title": "No sGO",
                "color": "cyan",
                "name" : "tgi_no_sgo"
            },
            {
                "title": "No SMF",
                "color": "orange",
                "name" : "tgi_no_smf"
            },
        ],
        "ylim" : [0,25],
        "aspect" : 1
    }

    figure_feature_contribution.generate_figures(spec, 
        "../results/exp_mn_feature_contribution", 
        "../results/exp_mn_feature_contribution/figures/tgi_bacc_drop.png")


if __name__ == "__main__":
    main()
