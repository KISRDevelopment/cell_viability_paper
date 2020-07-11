import json 
import shlex
import subprocess
import os 
import pandas as pd 
import sys 
import glob 

import utils.eval_funcs 

SMF_LABELS = ['Lethal', 'Reduced', 'Normal']
GI_LABELS = ['Negative', 'Neutral', 'Positive', 'Supp']

def main(path, output_path):
    
    paths = [p for p in glob.glob("%s/*" % path) if os.path.isdir(p)]
    results_summary = []
    
    for results_path in paths:

        cols = []
        r = utils.eval_funcs.average_results(results_path)
        
        row = {
            "model" : os.path.basename(results_path),
            "bacc" : r['bacc'],
            "acc" : r['acc'],
            "log_prob" : r['log_prob']
        }
        cols = ['model', 'bacc', 'acc', 'log_prob']

        num_classes = len(r['per_class_f1'])
        labels = SMF_LABELS if num_classes == 3 else GI_LABELS

        for c in range(num_classes):
            row['%s_auc_roc' % labels[c]] = r['per_class_auc_roc'][c]
            row['%s_bacc' % labels[c]] = r['per_class_baccs'][c]
            row['%s_auc_pr' % labels[c]] = r['per_class_auc_pr'][c]
            cols.extend(['%s_auc_roc' % labels[c], '%s_bacc' % labels[c], '%s_auc_pr' % labels[c]])
        
        results_summary.append(row)

        print("Processed %s" % results_path)
    results_df = pd.DataFrame(results_summary)
    
    results_df = results_df.sort_values(['bacc'], ascending=False)

    results_df.to_excel(output_path, index=False, columns=cols, float_format='%0.2f')


if __name__ == "__main__":
    path = sys.argv[1]
    output_path = sys.argv[2]

    main(path, output_path)
