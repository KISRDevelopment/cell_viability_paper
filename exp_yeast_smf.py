import numpy as np 
import models.train_and_evaluate
import json 
import copy 

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


    #summarize_results("../results/exp_yeast_smf/cv_full", ['L', 'R', 'N'])
    #summarize_results("../results/exp_yeast_smf/cv_refined", ['L', 'R', 'N'])
    #summarize_results("../results/exp_yeast_smf/cv_mn", ['L', 'R', 'N'])
    #summarize_results("../results/exp_yeast_smf/cv_null", ['L', 'R', 'N'])


def run_cv_on_spec(model_spec, name, mode='cv'):
    models.train_and_evaluate.cv(model_spec, 
                                "../generated-data/dataset_yeast_smf.feather", 
                                "../generated-data/splits/dataset_yeast_smf_dev_test.npz",
                                mode,
                                "../results/exp_yeast_smf/%s_%s" % (mode, name),
                                n_workers=32,
                                no_train=False)


def summarize_results(model_output_path, class_labels):
    with open("%s/results.json" % model_output_path, 'r') as f:
        r = json.load(f)

        results = r['results']

        row = {}

        row['bacc'] = np.mean([e['bacc'] for e in results])
        row['acc'] = np.mean([e['acc'] for e in results])
        row['cm'] = np.mean([e['cm'] for e in results], axis=0)

        num_classes = len(class_labels)
            
        for c in range(num_classes):
            row['%s_auc_roc' % class_labels[c]] = np.mean([e['auc_roc'][c] for e in results])
            row['%s_bacc' % class_labels[c]] = np.mean([e['per_class_bacc'][c] for e in results])
            row['%s_auc_pr' % class_labels[c]] = np.mean([e['pr'][c] for e in results])
                
    print(row)

if __name__ == "__main__":
    main()
