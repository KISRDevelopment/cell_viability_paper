import sys
import numpy as np
import numpy.random as rng
import pandas as pd
import os
import sklearn.model_selection

SEED = 425345
rng = np.random.RandomState(SEED)

def main(dataset_path, reps, folds, valid_p, dev_test, test_p, output_path=None, **kwargs):

    df = pd.read_csv(dataset_path)
    all_genes = list(set(df['a_id']) | set(df['b_id']))
    print("Total # genes: %d" % len(all_genes))
    rng.shuffle(all_genes)

    if dev_test:
        n_dev = int((1-test_p) * len(all_genes))
        dev_genes = all_genes[:n_dev]
        test_genes = all_genes[n_dev:]
    else:
        dev_genes = all_genes
        test_genes = []
    
    print("Development set genes: %d" % len(dev_genes))
    print("Test set genes: %d" % len(test_genes))

    splits = []
    
    rkf = sklearn.model_selection.RepeatedKFold(n_splits=folds, 
        n_repeats=reps, 
        random_state=rng)
    
    dev_genes = np.array(dev_genes)
    
    i = 0

    for train_index, test_index in rkf.split(dev_genes):
        dev_test_genes = dev_genes[test_index]

        train_genes = dev_genes[train_index]
        rng.shuffle(train_genes)
        n_valid = int(valid_p * len(train_genes))
        valid_genes = train_genes[:n_valid]
        train_genes = train_genes[n_valid:]
        
        #print("Train genes: %d, Valid: %d, Dev Test: %d, Test: %d" % 
        #    (len(train_genes), len(valid_genes), len(dev_test_genes), len(test_genes)))

        splits.append({
            "train_genes" : train_genes,
            "valid_genes" : valid_genes,
            "dev_test_genes" : dev_test_genes,
            "test_genes" : test_genes
        })

        assert set(train_genes) & set(valid_genes) == set()
        assert set(train_genes) & set(dev_test_genes) == set()
        assert set(train_genes) & set(test_genes) == set()
        assert set(valid_genes) & set(dev_test_genes) == set()
        assert set(valid_genes) & set(test_genes) == set()
        assert set(dev_test_genes) & set(test_genes) == set()
        
    np.savez(output_path, splits=splits, 
                          reps=reps, 
                          folds=folds,
                          valid_p=valid_p,
                          dev_test=dev_test,
                          test_p=test_p)

if __name__ == "__main__":

    path = sys.argv[1]
    reps = int(sys.argv[2])
    folds = int(sys.argv[3])
    
    main(path, reps, folds, 0.2, False, 0.2, '../tmp/test')