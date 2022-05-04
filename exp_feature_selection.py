from math import comb
import numpy as np 
import itertools 
import json 
import copy 
import models.train_and_evaluate
import glob 
import pandas as pd 
import os 

SMF_LABELS = ['Lethal', 'Reduced', 'Normal']
GI_LABELS = ['Negative', 'Neutral', 'Positive', 'Supp']
TGI_LABELS = ['Negative', 'Neutral']

DEBUGGING = False # reduces number of combinations if true ... for testing purposes

def main():
    os.makedirs('../results/exp_feature_selection/smf', exist_ok=True)
    os.makedirs('../results/exp_feature_selection/gi', exist_ok=True)
    os.makedirs('../results/exp_feature_selection/tgi', exist_ok=True)

    smf_combinations = make_smf_combinations()
    brute_force_fs(smf_combinations, 
        dataset_path='../generated-data/dataset_yeast_smf.feather', 
        splits_path='../generated-data/splits/dataset_yeast_smf_dev_test.npz',
        output_path='../results/exp_feature_selection/smf')
    smf_df = summarize_results('../results/exp_feature_selection/smf', SMF_LABELS)
    smf_df.to_excel("../results/exp_feature_selection/smf.xlsx", index=False)

    gi_combinations = make_gi_combinations("cfgs/gi_nn_model.json")
    brute_force_fs(gi_combinations, 
        dataset_path='../generated-data/dataset_yeast_gi_hybrid.feather', 
        splits_path='../generated-data/splits/dataset_yeast_gi_hybrid_dev_test.npz',
        output_path='../results/exp_feature_selection/gi',
        sg_path="../generated-data/dataset_yeast_allppc.feather", n_workers=10)
    gi_df = summarize_results('../results/exp_feature_selection/gi', GI_LABELS)
    gi_df.to_excel("../results/exp_feature_selection/gi.xlsx", index=False)

    tgi_combinations = make_gi_combinations("cfgs/tgi_nn_model.json")
    brute_force_fs(tgi_combinations, 
        dataset_path='../generated-data/dataset_yeast_tgi.feather', 
        splits_path='../generated-data/splits/dataset_yeast_tgi_dev_test.npz',
        output_path='../results/exp_feature_selection/tgi',
        sg_path="../generated-data/dataset_yeast_allppc.feather", n_workers=10)
    tgi_df = summarize_results('../results/exp_feature_selection/tgi', TGI_LABELS)
    tgi_df.to_excel("../results/exp_feature_selection/tgi.xlsx", index=False)
def brute_force_fs(model_specs, dataset_path, splits_path, output_path, n_workers=32, **kwargs):
    if DEBUGGING:
        model_specs = model_specs[:5]
    
    model_output_paths = ["%s/%s" % (output_path, '-'.join(m['name'])) for m in model_specs]
    
    models.train_and_evaluate.multiple_cv(model_specs, 
        model_output_paths, 
        dataset_path, 
        splits_path, 
        "cv", 
        n_workers=n_workers, 
        no_train=False, 
        skip_if_exists=True,
        **kwargs)

def summarize_results(results_path, class_labels):

    dirs = glob.glob(results_path + "/*")

    summary = []
    for directory in dirs:

        with open("%s/results.json" % directory, 'r') as f:
            r = json.load(f)

            model_spec = r['model_spec']
            results = r['results']

            row = {}

            row['bacc'] = np.mean([e['bacc'] for e in results])
            row['acc'] = np.mean([e['acc'] for e in results])
            
            num_classes = len(class_labels)
            
            for c in range(num_classes):
                row['%s_auc_roc' % class_labels[c]] = np.mean([e['auc_roc'][c] for e in results])
                row['%s_bacc' % class_labels[c]] = np.mean([e['per_class_bacc'][c] for e in results])
                row['%s_auc_pr' % class_labels[c]] = np.mean([e['pr'][c] for e in results])
            

            row['name'] = '-'.join(model_spec['name'])

            summary.append(row)
        

    df = pd.DataFrame(summary)

    df = df.sort_values(['bacc'], ascending=False)

    return df

def make_smf_combinations():
    with open("cfgs/smf_nn_model.json", 'r') as f:
        model_spec = json.load(f)
    feature_groups = {
        "topology" : ["topology"],
        "sgo" : ["sgo"],
        "redundancy" : ["redundancy"],
        "phosphorylation" : ["phosphotase", "kinase"],
        "transcription" : ["transcription"],
        "abundance" : ["abundance_"],
        "localization" : ["localization_"]
    }
    
    fsets = list(model_spec['feature_sets'].keys())

    feature_groups = expand_feature_groups(fsets, feature_groups)

    keys = list(feature_groups.keys())

    combinations = list(powerset(keys))[1:]
    
    new_specs = []
    for i, combination in enumerate(combinations):
        
        new_model_spec = copy.deepcopy(model_spec)
        new_model_spec['name'] = sorted(combination)
        new_model_spec['selected_feature_sets'] = list(itertools.chain(*[feature_groups[g] for g in combination]) )
    
        new_specs.append(new_model_spec)
    
    return new_specs

def make_gi_combinations(spec_path):
    with open(spec_path, 'r') as f:
        model_spec = json.load(f)
    
    single_feature_groups = {
        "topology" : ["topology"],
        "sgo" : ["sgo"],
        "redundancy" : ["redundancy"],
        "phosphorylation-transcription" : ["phosphotase", "kinase", "transcription"],
        "abundance-localization" : ["abundance_", "localization_"],
        "smf" : ["smf"]
    }

    double_feature_groups = {
        "pairwise" : ["pairwise"]
    }

    fsets = list(model_spec['single_gene_spec']['feature_sets'].keys())
    single_feature_groups = expand_feature_groups(fsets, single_feature_groups)

    fsets = list(model_spec['double_gene_spec']['feature_sets'].keys())
    double_feature_groups = expand_feature_groups(fsets, double_feature_groups)
    
    sf_groups = set(single_feature_groups.keys())
    df_groups = set(double_feature_groups.keys())
    keys = list(sf_groups) + list(df_groups)

    combinations = list(powerset(keys))[1:]
    print("Number of combinations: %d" % len(combinations))
    new_specs = []
    for i, combination in enumerate(combinations):
        
        new_model_spec = copy.deepcopy(model_spec)
        new_model_spec['name'] = sorted(combination)

        combination_sg = [g for g in combination if g in single_feature_groups]
        combination_dg = [g for g in combination if g in double_feature_groups]
        
        new_model_spec['single_gene_spec']['selected_feature_sets'] = \
            list(itertools.chain(*[single_feature_groups[g] for g in combination_sg]) )
    
        new_model_spec['double_gene_spec']['selected_feature_sets'] = \
            list(itertools.chain(*[double_feature_groups[g] for g in combination_dg]) )

        # print(new_model_spec['name'])
        # print(new_model_spec['single_gene_spec']['selected_feature_sets'])
        
        # print(new_model_spec['double_gene_spec']['selected_feature_sets'])

        new_specs.append(new_model_spec)
    
    return new_specs

def expand_feature_groups(fsets, feature_groups):
    
    new_feature_groups = {}
    for fg, candidates in feature_groups.items():
        feature_sets = []
        for candidate in candidates:
            feature_sets.extend([fs for fs in fsets if fs.startswith(candidate)])
        new_feature_groups[fg] = list(set(feature_sets))
        assert len(new_feature_groups[fg]) > 0

    return new_feature_groups

# https://docs.python.org/2/library/itertools.html#recipes
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


if __name__ == "__main__":
    main()
