import pandas as pd
import numpy as np 

import sys 
import os
import json 

import sklearn.metrics
import scipy.stats as stats 
import numpy.random as rng
import uuid

import utils.eval_funcs
import keras.utils 
from termcolor import colored
def main(cfg, rep, fold, output_path, print_results=False):
    

    
    dataset_path = cfg['task_path']
    targets_path = cfg['targets_path']
    train_test_path = cfg['splits_path']

    # load dataset
    df = pd.read_csv(dataset_path)
    
    # create output
    Y = keras.utils.to_categorical(np.load(targets_path)['y'])
    
    # load train/test split 
    data = np.load(train_test_path)
    train_sets = data['train_sets']
    valid_sets = data['valid_sets']
    test_sets = data['test_sets']

    # create training and testing data frames
    train_ix = train_sets[rep, fold,:]
    valid_ix = valid_sets[rep,fold,:]
    test_ix = test_sets[rep,fold,:]

    if cfg.get("train_on_full_dataset", False):
        print(colored("******** TRAINING ON FULL DATASET ***********", "red"))
        train_ix = train_ix + test_ix
    
    if cfg.get("test_on_full_dataset", False):
        print(colored("******** TESTING ON FULL DATASET ***********", "green"))
        test_ix = train_ix + test_ix + valid_ix
    
    train_df = df.iloc[train_ix]
    valid_df = df.iloc[valid_ix]
    test_df = df.iloc[test_ix]

    train_Y = Y[train_ix,:]
    valid_Y = Y[valid_ix,:]
    test_Y = Y[test_ix, :]

    if cfg.get("train_model", True):
        props = np.mean(train_Y, axis=0, keepdims=True)
        if cfg.get("trained_model_path", None) is not None:
            print("Saving model")
            np.save(cfg["trained_model_path"], props)
    else:
        props = np.load(cfg["trained_model_path"] + ".npy")
    
    preds = np.tile(props, (test_Y.shape[0], 1))
    
    y_target = np.argmax(test_Y, axis=1)

    r, cm = utils.eval_funcs.eval_classifier(y_target, preds)
    
    utils.eval_funcs.print_eval_classifier(r)
    
    np.savez(output_path,
        preds = preds,
        y_target = y_target,
        cfg = cfg, 
        r=r,
        cm=cm,
        rep=rep,
        fold=fold)
    

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    rep = int(sys.argv[2])
    fold = int(sys.argv[3])
    output_path = sys.argv[4]

    # load model configuration --- it just needs dataset, targets, and splits paths
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    main(cfg, rep, fold, output_path)