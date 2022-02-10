import numpy as np 
import os 
import pandas as pd 
import sys 
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import glob 
from collections import defaultdict

COLS = ["is_neutral", "rel_not_ppc", "rel_not_phospho", "rel_not_trans"]

def main(path, output_path):

    result_df = average_results(path)

    result_df.to_csv(output_path, index=False)

    print(result_df) 

def average_results(cv_dir):

    
    files = get_files(cv_dir)

    task_df = None 
    test_splits = None 

    result_rows = []
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

        row = {}
        for col in COLS:
            ytrue = sdf[col]

            bacc = balanced_accuracy_score(ytrue, ypred)
            roc = roc_auc_score(ytrue, d['preds'][:,1])

            print("   %s: %0.2f, %0.2f" % (col, bacc, roc))
            row["bacc_%s"%col] = bacc 
            row["roc_%s"%col] = roc 

        result_rows.append(row)
    
    return pd.DataFrame(result_rows)

def get_files(cv_dir):
    if cv_dir.endswith('.npz'):
        return [cv_dir]
    return glob.glob("%s/*" % cv_dir)
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])