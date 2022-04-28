import numpy as np 
import itertools 
import json 
import copy 
import models.train_and_evaluate
import glob 
import pandas as pd 

SMF_LABELS = ['Lethal', 'Reduced', 'Normal']
GI_LABELS = ['Negative', 'Neutral', 'Positive', 'Supp']

DEBUGGING = True # reduces number of combinations if true ... for testing purposes

def main():

    # optimize_hyperparams('cfgs/smf_nn_model.json', 
    #                      '../generated-data/dataset_yeast_smf.feather', 
    #                      '../generated-data/splits/dataset_yeast_smf_dev_test.npz',
    #                      '../results/smf_nn_model_hyperparam_opt')
    # smf_df = summarize_results('../results/smf_nn_model_hyperparam_opt', SMF_LABELS)

    optimize_hyperparams('cfgs/gi_nn_model.json', 
                         '../generated-data/dataset_yeast_gi_hybrid.feather', 
                         '../generated-data/splits/dataset_yeast_gi_hybrid_dev_test.npz',
                         '../results/gi_nn_model_hyperparam_opt',
                         sg_path='../generated-data/dataset_yeast_allppc.feather', n_workers=16)
    gi_df = summarize_results('../results/gi_nn_model_hyperparam_opt', GI_LABELS)

    writer = pd.ExcelWriter('../results/hyperparam_opt_results.xlsx')
    smf_df.to_excel(writer, sheet_name='S-Full', index=False)
    gi_df.to_excel(writer, sheet_name='D-Full', index=False)
    writer.save()

def optimize_hyperparams(model_spec_path, dataset_path, splits_path, output_path, n_workers=32, **kwargs):

    with open(model_spec_path, 'r') as f:
        model_spec = json.load(f)

    model_specs = make_hyperparam_combinations(model_spec)
    if DEBUGGING:
        model_specs = model_specs[:5]
    
    model_output_paths = ["%s/comb%d" % (output_path, i) for i in range(len(model_specs))]
    
    models.train_and_evaluate.multiple_cv(model_specs, model_output_paths, dataset_path, splits_path, "cv", no_train=False, n_workers, **kwargs)

def summarize_results(results_path, class_labels):

    dirs = glob.glob(results_path + "/*")

    summary = []
    for directory in dirs:

        with open("%s/results.json" % directory, 'r') as f:
            r = json.load(f)

            model_spec = r['model_spec']
            results = r['results']

            row = model_spec['chosen_hyperparams']

            row['bacc'] = np.mean([e['bacc'] for e in results])
            row['acc'] = np.mean([e['acc'] for e in results])

            
            num_classes = len(class_labels)
            
            for c in range(num_classes):
                row['%s_auc_roc' % class_labels[c]] = np.mean([e['auc_roc'][c] for e in results])
                row['%s_bacc' % class_labels[c]] = np.mean([e['per_class_bacc'][c] for e in results])
                row['%s_auc_pr' % class_labels[c]] = np.mean([e['pr'][c] for e in results])
                
            summary.append(row)

    df = pd.DataFrame(summary)
    df = df.sort_values(['bacc'], ascending=False)

    return df

def make_hyperparam_combinations(model_spec):

    keys = list(model_spec['hyperparams'].keys())
    vals = [model_spec['hyperparams'][k] for k in keys]

    combinations = [dict(zip(keys, comb)) for comb in itertools.product(*vals)]

    new_specs = []
    for i, combination in enumerate(combinations):
        
        new_model_spec = copy.deepcopy(model_spec)

        new_model_spec['chosen_hyperparams'] = combination

        if new_model_spec['class'] == 'nn_single':
            subspecs = [new_model_spec]
        elif new_model_spec['class'] == 'nn_double':
            subspecs = [new_model_spec['single_gene_spec'], new_model_spec['double_gene_spec']]
        
        for key, value in combination.items():
            if key == 'learning_rate':
                new_model_spec['learning_rate'] = value 
            elif key == 'embedding_size':
                for subspec in subspecs:
                    subspec['embedding_size'] = value 
            elif key == 'layer_sizes':
                for subspec in subspecs:
                    for module, module_props in subspec['modules'].items():
                        module_props['layer_sizes'] = combination['layer_sizes']
            else:
                raise Exception("Unrecognized hyperparameter %s" % key)
        new_specs.append(new_model_spec)
    
    return new_specs

if __name__ == "__main__":
    main()
