import json 
import pandas as pd
import numpy as np
import models.nn_single
import models.nn_double
import models.nn_triple
import models.mn 
import models.common

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

def train_and_evaluate(model_spec, df, split, split_mode, model_output_path, sg_path=None):

    model_class = MODELS[model_spec['class']]

    m = model_class(model_spec, sg_path=sg_path)
    
    train_df, valid_df, test_df = models.common.get_dfs(df, split, **SPLIT_MODES[split_mode])
    
    m.train(train_df, valid_df)

    m.save(model_output_path)

    preds = m.predict(test_df)

    target_col = model_spec['target_col']
    r = models.common.evaluate(np.array(test_df[target_col]), preds)

    return r 

def main(model_spec_path, dataset_path, splits_path, split_id, split_mode, model_output_path, sg_path=None):

    with open(model_spec_path, 'r') as f:
        model_spec = json.load(f)
    
    df = pd.read_feather(dataset_path)

    splits = np.load(splits_path, allow_pickle=True)['splits']
    split = splits[int(split_id)]
    
    r = train_and_evaluate(model_spec, df, split, split_mode, model_output_path, sg_path)

    return r 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_spec_path", type=str, help="Path to json file specifying model configuration.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset .feather file.")
    parser.add_argument("splits_path", type=str, help="Path to the splits file which specifies the training/validation/testing splits.")
    parser.add_argument("split_id", type=int, help="Split number in the splits file.")
    parser.add_argument("split_mode", type=str, choices=SPLIT_MODES.keys(), help="Split mode.")
    parser.add_argument("model_output_path", type=str, help="Where to store the trained model.")
    parser.add_argument("--sg_path", type=str, help="Path to file containing features for single genes (relevant for double and triple nn models).")

    args = vars(parser.parse_args())
    
    main(**args)