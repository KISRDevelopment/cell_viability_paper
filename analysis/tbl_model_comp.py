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

FMAP = {
    "Go" : "sGO"
}

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
        model_name = os.path.basename(results_path)
        
        if os.path.isdir(results_path):
            cfg = np.load(results_path + '/run_0_0.npz', allow_pickle=True)['cfg'].item()
        else:
            cfg = np.load(results_path, allow_pickle=True)['cfg'].item()
        # get feature groups
        included_features = [s['name'] for s in cfg['spec']]
        feature_groups = [c.capitalize() for c in 
            sorted(set([features_to_groups[f] for f in included_features if f in features_to_groups]))]
        feature_groups = [FMAP.get(fname, fname) for fname in feature_groups]
        num_features = len(feature_groups)

        # get specific selections
        selected_features = []
        has_full_pairwise = len([s for s in cfg['spec'] if s.get('pairwise', False) and s['name'] == 'pairwise']) > 0

        for s in cfg['spec']:
            if s.get('pairwise', False) and not has_full_pairwise:
                if s['name'] not in ('pairwise', 'pairwise_const'):
                    selected_features.append(s['name'])
            else:
                sf = s.get('selected_features', [])
                if sf is not None:
                    selected_features.extend(sf)
                if s.get('feature', None) is not None:
                    selected_features.append(s['feature'])
        row = {
            "model" : ", ".join(feature_groups),
            "no. features" : num_features,
            "selected sub-features" : ", ".join(selected_features),
            "filename" : model_name,
            "bacc" : r['bacc'],
            "acc" : r['acc']
        }
        cols = ['model', "filename", 'no. features', 'selected sub-features', 'bacc', 'acc']

        num_classes = len(r['per_class_f1'])
        #labels = SMF_LABELS if num_classes == 3 else GI_LABELS

        for c in range(num_classes):
            row['%s_auc_roc' % labels[c]] = r['per_class_auc_roc'][c]
            row['%s_bacc' % labels[c]] = r['per_class_baccs'][c]
            row['%s_auc_pr' % labels[c]] = r['per_class_auc_pr'][c]
            cols.extend(['%s_auc_roc' % labels[c], '%s_bacc' % labels[c], '%s_auc_pr' % labels[c]])
        
        results_summary.append(row)

        print("... done")
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
