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

FMAP = {
    "Go" : "sGO"
}

def main(path, output_path):
    
    paths = [p for p in glob.glob("%s/*" % path) if os.path.isdir(p)]
    results_summary = []
    
    for results_path in paths:

        cols = []
        r = utils.eval_funcs.average_results(results_path)
        model_name = os.path.basename(results_path)
        row = {
            "model" : make_friendly_model_name(model_name),
            "no. features" : len(model_name.split('~')),
            "bacc" : r['bacc'],
            "acc" : r['acc']
        }
        cols = ['model', 'no. features', 'bacc', 'acc']

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

def make_friendly_model_name(s):
    
    def fs_with_selected_feature(f):
        parts = f.split('--')
        if len(parts) > 1:
            return "%s (%s)" % (parts[0].capitalize(), parts[1].capitalize())
        else: 
            fname = f.capitalize()
            return FMAP.get(fname, fname)
    
    features = [fs_with_selected_feature(f) for f in s.split('~')]
    return ", ".join(features)

if __name__ == "__main__":
    path = sys.argv[1]
    output_path = sys.argv[2]

    main(path, output_path)
