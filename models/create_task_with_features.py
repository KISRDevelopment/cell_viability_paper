import numpy as np 
import pandas as pd 
import sys 
import json
import models.feature_loader as feature_loader
import scipy.stats as stats
def main(cfg_path):

    dataset_path = cfg['task_path']
    
    # load dataset
    df = pd.read_csv(dataset_path)
    
    # load input features
    single_gene_spec = [s for s in cfg['spec'] if not s['pairwise']]
    pairwise_gene_spec = [s for s in cfg['spec'] if s['pairwise']]
    single_fsets, single_fsets_shapes = feature_loader.load_feature_sets(single_gene_spec, False)
    pairwise_fsets, pairwise_fsets_shapes = feature_loader.load_feature_sets(pairwise_gene_spec, False)
    
    inputs = feature_transform(df, single_fsets, pairwise_fsets)
    diff_features = feature_dist(inputs[0], inputs[1])
    diff_features = stats.zscore(diff_features, axis=0)
    inputs.append(diff_features)

    X = np.hstack(inputs)

    output_path = cfg['features_path']
    np.save(output_path, X)
    
def feature_dist(a, b):

    x = a + 0.000000001
    y = b + 0.000000001
    diff = x - y 
    avg = (x + y) / 2
    return diff / avg 

def feature_transform(df, single_fsets, pairwise_fsets):
    inputs_A = []
    inputs_B = []
    inputs_AB = []
    for fset in single_fsets:
        inputs_A.append(fset[df['a_id'], :])
        inputs_B.append(fset[df['b_id'], :])

    a_id = np.array(df['a_id'])
    b_id = np.array(df['b_id'])


    for fset in pairwise_fsets:

        if hasattr(fset, 'transform'):
            inputs_AB.append(fset.transform(df))
        
        elif type(fset) != dict:
            inputs_AB.append(fset[df['a_id'], df['b_id'], :])

        else:
            first_val = next(iter(fset.values()))
            fset_shape = len(first_val)

            PF = []
            for i in range(df.shape[0]):
                key = tuple(sorted((a_id[i], b_id[i])))
                if key in fset:
                    PF.append(fset[key])
                else:
                    PF.append(np.zeros(fset_shape))
            PF = np.array(PF)
            inputs_AB.append(PF)
            
    return inputs_A + inputs_B + inputs_AB


if __name__ == "__main__":
    
    cfg_path = sys.argv[1]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    main(cfg)