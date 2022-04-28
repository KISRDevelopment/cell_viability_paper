import json 
import pandas as pd
import numpy as np
import models.nn_single
import models.nn_double
import models.nn_triple
import models.mn 
import models.common
import multiprocessing 
import os 

MODELS = {
    "nn_single" : models.nn_single.SingleInputNNModel,
    "nn_double" : models.nn_double.DoubleInputNNModel,
    "nn_triple" : models.nn_triple.TripleInputNNModel,
    "mn" : models.mn.MnModel
}

SPLIT_MODES = {
    "cv": {
        "train_ids" : [1],
        "valid_ids" : [2],
        "test_ids" : [3]
    },
    "dev_test": {
        "train_ids" : [1, 3],
        "valid_ids" : [2],
        "test_ids" : [0]
    },
    "full" : { # used for generalization experiments (test is irrelevant here)
        "train_ids" : [0, 1, 3],
        "valid_ids" : [2],
        "test_ids" : [2]
    }
}

# global variable that are shared among processes (in unix)
# this is to avoid loading a large dataset into memory in every subprocess during CV
df = None
splits = None 

def single_split(model_spec, df, splits, split_id, split_mode, model_output_path, sg_path=None, no_train=True):

    model_class = MODELS[model_spec['class']]
    split = splits[split_id]

    train_df, valid_df, test_df = models.common.get_dfs(df, split, **SPLIT_MODES[split_mode])
    
    if no_train:
        m = model_class.load(model_output_path)
    else:
        m = model_class(model_spec, sg_path=sg_path)    
        m.train(train_df, valid_df)
        m.save(model_output_path)
        
    
    preds = m.predict(test_df)

    target_col = model_spec['target_col']
    r = models.common.evaluate(np.array(test_df[target_col]), preds)
    r['split_id'] = split_id

    print("Done %s %d" % (model_output_path, split_id))

    return r 

def cv_f(task):
    global df 
    global splits 

    task['df'] = df 
    task['splits'] = splits 

    r = single_split(**task)
    
    return r 

def cv(model_spec_path, dataset_path, splits_path, split_mode, model_output_path, sg_path=None, n_workers=4, no_train=True, **kwargs):
    global df 
    global splits 

    os.makedirs(model_output_path, exist_ok=True)

    d = np.load(splits_path, allow_pickle=True)
    n_splits = len(d['splits'])

    if split_mode == 'dev_test':
        n_splits = 1
    
    with open(model_spec_path, 'r') as f:
        model_spec = json.load(f)
    model_spec['verbose'] = False 

    df = pd.read_feather(dataset_path)

    splits = np.load(splits_path, allow_pickle=True)['splits']

    tasks = [{ "model_spec" : model_spec, 
               "split_id" : i,
               "split_mode" : split_mode, 
               "model_output_path" : "%s/model%d.npz" % (model_output_path, i), 
               "sg_path" : sg_path,
               "no_train" : no_train } for i in range(n_splits)]
    
    if n_workers == 1:
        for task in tasks:
            cv_f(task)
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(cv_f, tasks)
                
    
    with open("%s/results.json" % model_output_path, "w") as f:
        json.dump({ "model_spec" : model_spec, "results" : results }, f, indent=4)

    return results 


def multiple_cv(model_specs, model_output_paths, dataset_path, splits_path, split_mode, sg_path=None, n_workers=4, no_train=True, **kwargs):
    """ 
        Optimized processor utilization when running same dataset CV with multiple specs (like in feature selection and hyperparam opt) 
        This is done by feeding the multiprocessing pool all the CV tasks for all model specs at once.
    """
    global df 
    global splits 

    d = np.load(splits_path, allow_pickle=True)
    n_splits = len(d['splits'])

    if split_mode == 'dev_test':
        n_splits = 1
    
    df = pd.read_feather(dataset_path)

    splits = np.load(splits_path, allow_pickle=True)['splits']

    tasks = []
    for model_spec, model_output_path in zip(model_specs, model_output_paths):   
        os.makedirs(model_output_path, exist_ok=True)
        model_spec['verbose'] = False 

        tasks.extend([{ "model_spec" : model_spec, 
                "split_id" : i,
                "split_mode" : split_mode, 
                "model_output_path" : "%s/model%d.npz" % (model_output_path, i), 
                "sg_path" : sg_path,
                "no_train" : no_train } for i in range(n_splits)])
    
    print("Total tasks: %d" % len(tasks))
    
    if n_workers == 1:
        for task in tasks:
            cv_f(task)
    else:
        with multiprocessing.Pool(processes=n_workers,maxtasksperchild=1) as pool:
            results = pool.map(cv_f, tasks, chunksize=1) # chunk size controls the number of tasks
                
    offset = 0
    for model_spec, model_output_path in zip(model_specs, model_output_paths):
        results_subset = results[offset:(offset+n_splits)]
    
        offset += n_splits
        with open("%s/results.json" % model_output_path, "w") as f:
            json.dump({ "model_spec" : model_spec, "results" : results_subset }, f, indent=4)

    return results 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_spec_path", type=str, help="Path to json file specifying model configuration.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset .feather file.")
    parser.add_argument("splits_path", type=str, help="Path to the splits file which specifies the training/validation/testing splits.")
    parser.add_argument("split_mode", type=str, choices=SPLIT_MODES.keys(), help="Split mode. If dev_test, only one split will be run.")
    parser.add_argument("model_output_path", type=str, help="Where to store the trained model. For CV, this should be a path to a directory.")
    parser.add_argument("--sg_path", type=str, help="Path to file containing features for single genes (relevant for double and triple nn models).")
    parser.add_argument("--split_id", type=int, default=-1, help="Run only the given split number in the splits file. Otherwise, CV will be done.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of worker processes to use when doing CV.")
    parser.add_argument("--no-train", action='store_true', default=False, help="Train the model or evaluate pre-traiend models on the test set.")

    args = vars(parser.parse_args())

    if args['split_id'] > -1:
        single_split(**args)
    else:
        cv(**args)