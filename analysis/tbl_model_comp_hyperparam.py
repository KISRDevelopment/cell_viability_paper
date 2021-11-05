import json 
import shlex
import subprocess
import os 
import pandas as pd 
import sys 
import glob 
from utils.features_to_groups import features_to_groups
import utils.eval_funcs 
import numpy as np

SMF_LABELS = ['Lethal', 'Reduced', 'Normal']
GI_LABELS = ['Negative', 'Neutral', 'Positive', 'Supp']

def main(path, output_path, labels, use_glob_spec=False):
    
    if use_glob_spec:
        glob_spec = path 
        paths = [p for p in glob.glob(glob_spec)]
    else:
        glob_spec = "%s/*" % path 
        paths = [p for p in glob.glob(glob_spec) if os.path.isdir(p)]
    
    results_summary = []
    
    for results_path in paths:
        print("Processing %s" % results_path, end='')
        cols = []
        r = utils.eval_funcs.average_results(results_path)
        
        if os.path.isdir(results_path):
            cfg = np.load(results_path + '/run_0_0.npz', allow_pickle=True)['cfg'].item()
        else:
            cfg = np.load(results_path, allow_pickle=True)['cfg'].item()
        
        
        row = {
            "layer_sizes" : cfg['layer_sizes'],
            "embedding_size" : cfg['embedding_size'],
            "learning_rate" : cfg['learning_rate'],
            "bacc" : r['bacc'],
            "acc" : r['acc']
        }
        cols = ['layer_sizes', "embedding_size", 'learning_rate', 'bacc', 'acc']

        num_classes = len(r['per_class_f1'])
        
        for c in range(num_classes):
            row['%s_auc_roc' % labels[c]] = r['per_class_auc_roc'][c]
            row['%s_bacc' % labels[c]] = r['per_class_baccs'][c]
            row['%s_auc_pr' % labels[c]] = r['per_class_auc_pr'][c]
            cols.extend(['%s_auc_roc' % labels[c], '%s_bacc' % labels[c], '%s_auc_pr' % labels[c]])
        
        results_summary.append(row)

        print("... done")
    results_df = pd.DataFrame(results_summary)
    
    results_df = results_df.sort_values(['bacc'], ascending=False)

    results_df.to_excel(output_path, index=False, columns=cols)

if __name__ == "__main__":
    path = sys.argv[1]
    output_path = sys.argv[2]

    main(path, output_path)
