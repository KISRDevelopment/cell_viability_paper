import json 
import pandas as pd
import numpy as np
import models.nn_single
import models.nn_double
import models.nn_triple
import models.mn 
import models.common
import multiprocessing 

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

def single_split(model_spec_path, dataset_path, splits_path, split_id, split_mode, model_output_path, sg_path=None, verbose=True):

    with open(model_spec_path, 'r') as f:
        model_spec = json.load(f)
    model_spec['verbose'] = verbose 
    
    df = pd.read_feather(dataset_path)

    splits = np.load(splits_path, allow_pickle=True)['splits']
    split = splits[int(split_id)]
    
    model_class = MODELS[model_spec['class']]

    m = model_class(model_spec, sg_path=sg_path)
    
    train_df, valid_df, test_df = models.common.get_dfs(df, split, **SPLIT_MODES[split_mode])
    
    m.train(train_df, valid_df)

    m.save(model_output_path)

    preds = m.predict(test_df)

    target_col = model_spec['target_col']
    r = models.common.evaluate(np.array(test_df[target_col]), preds)
    r['split_id'] = split_id

    return r 

def cv_f(packed):
    return single_split(*packed)

def cv(model_spec_path, dataset_path, splits_path, split_mode, model_output_path, sg_path=None, n_workers=4, **kwargs):

    d = np.load(splits_path)
    n_splits = len(d['splits'])

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(cv_f, 
            [(model_spec_path, dataset_path, splits_path, i, split_mode, "%s/model%d.npz" % (model_output_path, i), sg_path, False) for i in range(n_splits)])
    
    with open("%s/results.json" % model_output_path, "w") as f:
        json.dump(results, f, indent=4)

    return results 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_spec_path", type=str, help="Path to json file specifying model configuration.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset .feather file.")
    parser.add_argument("splits_path", type=str, help="Path to the splits file which specifies the training/validation/testing splits.")
    parser.add_argument("split_mode", type=str, choices=SPLIT_MODES.keys(), help="Split mode.")
    parser.add_argument("model_output_path", type=str, help="Where to store the trained model. For CV, this should be a path to a directory.")
    parser.add_argument("--sg_path", type=str, help="Path to file containing features for single genes (relevant for double and triple nn models).")
    parser.add_argument("--split_id", type=int, default=-1, help="Run only the given split number in the splits file. Otherwise, CV will be done.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of worker processes to use when doing CV.")

    args = vars(parser.parse_args())

    if args['split_id'] > -1:
        single_split(**args)
    else:
        cv(**args)