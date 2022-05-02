from genericpath import exists
import numpy as np 
import models.train_and_evaluate
import json 
import copy 
import os 
import pandas as pd 
import figure_cv_bacc
import figure_cm
import figure_auc_roc_curve
import models.mn 
import models.null
import concurrent.futures

def load_spec(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    mn_spec = load_spec("cfgs/smf_mn_model.json")
    mn_spec['target_col'] = 'is_viable'
    mn_spec['batch_size_p'] = 0.1

    model_output_paths = train_smf_model(mn_spec, "../results/exp_generalization/smf_model")
    evaluate(mn_spec, model_output_paths, "../generated-data/dataset_pombe_smf.feather", "../results/exp_generalization/s-mn_pombe.json")
    evaluate(mn_spec, model_output_paths, "../generated-data/dataset_human_smf.feather", "../results/exp_generalization/s-mn_human.json")
    evaluate(mn_spec, model_output_paths, "../generated-data/dataset_dro_smf.feather", "../results/exp_generalization/s-mn_dro.json")
    
    null_spec = { 'target_col' : 'is_viable', 'class' : 'null' }
    model_output_paths = train_smf_model(null_spec, "../results/exp_generalization/s-null_model")
    evaluate(null_spec, model_output_paths, "../generated-data/dataset_pombe_smf.feather", "../results/exp_generalization/s-null_pombe.json")
    evaluate(null_spec, model_output_paths, "../generated-data/dataset_human_smf.feather", "../results/exp_generalization/s-null_human.json")
    evaluate(null_spec, model_output_paths, "../generated-data/dataset_dro_smf.feather", "../results/exp_generalization/s-null_dro.json")
    
    generate_smf_figures('pombe')
    generate_smf_figures('human')
    generate_smf_figures('dro')

    mn_spec = load_spec("cfgs/gi_mn_model.json")
    mn_spec['target_col'] = 'is_neutral'
    model_output_paths = train_gi_model(mn_spec, "../results/exp_generalization/gi_model")
    evaluate(mn_spec, model_output_paths, "../generated-data/dataset_pombe_gi_mn.feather", "../results/exp_generalization/d-mn_pombe.json")
    evaluate(mn_spec, model_output_paths, "../generated-data/dataset_human_gi_mn.feather", "../results/exp_generalization/d-mn_human.json")
    evaluate(mn_spec, model_output_paths, "../generated-data/dataset_dro_gi_mn.feather", "../results/exp_generalization/d-mn_dro.json")
    
    mn_spec = load_spec("cfgs/gi_mn_model.json")
    mn_spec['target_col'] = 'is_neutral'
    mn_spec['features'].remove('sgo-')
    model_output_paths = train_gi_model(mn_spec, "../results/exp_generalization/gi_model_no_sgo")
    evaluate(mn_spec, model_output_paths, "../generated-data/dataset_pombe_gi_mn.feather", "../results/exp_generalization/d-mn_no_sgo_pombe.json")
    evaluate(mn_spec, model_output_paths, "../generated-data/dataset_human_gi_mn.feather", "../results/exp_generalization/d-mn_no_sgo_human.json")
    evaluate(mn_spec, model_output_paths, "../generated-data/dataset_dro_gi_mn.feather", "../results/exp_generalization/d-mn_no_sgo_dro.json")
    
    null_spec = { 'target_col' : 'is_neutral', 'class' : 'null' }
    model_output_paths = train_gi_model(null_spec, "../results/exp_generalization/d-null")
    evaluate(null_spec, model_output_paths, "../generated-data/dataset_pombe_gi_mn.feather", "../results/exp_generalization/d-null_pombe.json")
    evaluate(null_spec, model_output_paths, "../generated-data/dataset_human_gi_mn.feather", "../results/exp_generalization/d-null_human.json")
    evaluate(null_spec, model_output_paths, "../generated-data/dataset_dro_gi_mn.feather", "../results/exp_generalization/d-null_dro.json")
    
    generate_gi_figures('pombe')
    generate_gi_figures('human')
    generate_gi_figures('dro')

def train_smf_model(mn_spec, trained_model_path):
    
    df = pd.read_feather('../generated-data/dataset_yeast_smf.feather')
    d = np.load("../generated-data/splits/dataset_yeast_smf_dev_test.npz", allow_pickle=True)

    splits = d['splits']
    n_reps = d['reps']
    n_folds = d['folds']

    split_ids = [i * n_folds for i in range(n_reps)]

    return train(mn_spec, df, splits, split_ids, trained_model_path)

def train_gi_model(mn_spec, trained_model_path):
    df = pd.read_feather('../generated-data/dataset_yeast_gi_hybrid_mn.feather')
    d = np.load("../generated-data/splits/dataset_yeast_gi_hybrid_dev_test.npz", 
        allow_pickle=True)

    splits = d['splits']
    n_reps = d['reps']
    n_folds = d['folds']

    split_ids = [i * n_folds for i in range(n_reps)]

    return train(mn_spec, df, splits, split_ids, trained_model_path)

def train(model_spec, df, splits, split_ids, output_path):
    os.makedirs(output_path, exist_ok=True)
    model_spec['verbose'] = False 

    futures = []
    model_output_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(split_ids)) as executor:
        for i, split_id in enumerate(split_ids):
            train_df, valid_df, _ = models.common.get_dfs(df, splits[split_id], train_ids=[0, 1, 3], valid_ids=[2], test_ids=[0])
            model_output_path = os.path.join(output_path, 'model_%d.npz' % i)
            model_output_paths.append(model_output_path)
            futures.append(executor.submit(_train_model, model_spec, train_df, valid_df, model_output_path))
        
    concurrent.futures.wait(futures)

    return model_output_paths
    
def _train_model(model_spec, train_df, valid_df, output_path):
    if model_spec['class'] == 'mn':
        model = models.mn.MnModel(model_spec)
    else:
        model = models.null.NullModel(model_spec)
    model.train(train_df, valid_df)
    model.save(output_path)

def evaluate(model_spec, model_output_paths, test_path, results_path):
    model_class = models.mn.MnModel if model_spec['class'] == 'mn' else models.null.NullModel

    test_df = pd.read_feather(test_path)

    preds = []
    for model_output_path in model_output_paths:
        model = model_class.load(model_output_path)
        preds.append(model.predict(test_df, training_norm=False))
    preds = np.array(preds)
    preds = np.mean(preds, axis=0)

    target_col = model_spec['target_col']
    r = models.common.evaluate(np.array(test_df[target_col]), preds)

    with open(results_path, "w") as f:
        json.dump({ "model_spec" : model_spec, "results" : [r] }, f, indent=4)

def generate_smf_figures(org):
    output_dir = "../results/exp_generalization/figures/smf/%s" % org
    os.makedirs(output_dir, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "S-MN",
                "color": "#3A90FF",
                "name" : "s-mn_%s" % org,
                "results_path" : "../results/exp_generalization/s-mn_%s.json" % org
            },
            {
                "title": "Null",
                "color": "#c9c9c9",
                "star_color": "grey",
                "cm_color": "grey",
                "name" : "s-null_%s" % org,
                "results_path" : "../results/exp_generalization/s-null_%s.json" % org
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

    figure_cv_bacc.generate_figures(spec, "../results/exp_generalization", os.path.join(output_dir, 'overall_bacc.png'))

    for model in spec['models']:
        figure_cm.plot_cm(model['results_path'], model['color'], spec['short_classes'], os.path.join(output_dir, "cm_%s.png" % model['name']))
    
    for i in range(len(spec['classes'])):
        figure_auc_roc_curve.plot_auc_roc_curves(spec, i, os.path.join(output_dir, "auc_roc%s.png" % spec["short_classes"][i]))
    
def generate_gi_figures(org):
    output_dir = "../results/exp_generalization/figures/gi/%s" % org
    os.makedirs(output_dir, exist_ok=True)

    spec = {
        "models": [
            {
                "title": "D-MN",
                "color": "#3A90FF",
                "name" : "d-mn_%s" % org,
                "results_path" : "../results/exp_generalization/d-mn_%s.json" % org
            },
            {
                "title": "D-MN No sGO",
                "color": "#38fffc",
                "name" : "d-mn_no_sgo_%s" % org,
                "fsize" : 50,
                "results_path" : "../results/exp_generalization/d-mn_no_sgo_%s.json" % org
            },
            {
                "title": "Null",
                "color": "#c9c9c9",
                "star_color": "grey",
                "cm_color": "grey",
                "name" : "d-null_%s" % org,
                "results_path" : "../results/exp_generalization/d-null_%s.json" % org
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

    figure_cv_bacc.generate_figures(spec, "../results/exp_generalization", os.path.join(output_dir, 'overall_bacc.png'))

    for model in spec['models']:
        figure_cm.plot_cm(model['results_path'], model['color'], spec['short_classes'], os.path.join(output_dir, "cm_%s.png" % model['name']))
    
    for i in range(len(spec['classes'])):
        figure_auc_roc_curve.plot_auc_roc_curves(spec, i, os.path.join(output_dir, "auc_roc%s.png" % spec["short_classes"][i]))
    

if __name__ == "__main__":
    main()
