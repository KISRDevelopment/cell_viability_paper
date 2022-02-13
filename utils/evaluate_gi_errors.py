import numpy as np 
import os 
import pandas as pd 
import sys 
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import glob 
from collections import defaultdict

import matplotlib.pyplot as plt
col1 = "is_neutral"
col2 = "rel_not_ppc"

col1_labels = ['GI', 'Neutral']
col2_labels = ['Copresp', 'Not Copresp']

def main(path, output_path):

    avg_errors = average_results(path)
    print(avg_errors)

    # a = np.sum(avg_errors[:2, :], axis=0)
    # b = np.sum(avg_errors[2:,:], axis=0)

    # a = a / np.sum(a)
    # b = b / np.sum(b)

    # print(a)
    # print(b)
    #result_df.to_csv(output_path, index=False)

    #print(result_df) 

def plot_matrix(errors):
    pass

def average_results(cv_dir):

    
    files = get_files(cv_dir)

    task_df = None 
    test_splits = None 

    all_results = []
    unnormed = []
    for file in files:
        print(file)
        d = np.load(file, allow_pickle=True)
        cfg = d['cfg'].item()

        if task_df is None:
            print("Loading task and splits")
            task_df = pd.read_csv(cfg['task_path'])
            splits = np.load(cfg['splits_path'])
            test_splits = splits['test_sets']
        
        rep, fold = d['rep'], d['fold']

        test_ix = test_splits[rep, fold, :]

        sdf = task_df[test_ix]
        
        ypred = np.argmax(d['preds'], axis=1)
        
        ytrue1 = sdf[col1]
        ytrue2 = sdf[col2]

        acc = np.array(ypred == ytrue1).astype(int) 
        
        combs = np.zeros((4, 2))
        for a, b, accr in zip(ytrue1, ytrue2, ypred):
            index = a * 2 + b
            combs[index,accr] += 1
        
        unnormed.append(combs)

        combs = combs / np.sum(combs, axis=1, keepdims=True)

        all_results.append(combs)
    
    return np.mean(all_results, axis=0), unnormed


def get_files(cv_dir):
    if cv_dir.endswith('.npz'):
        return [cv_dir]
    return glob.glob("%s/*" % cv_dir)
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])