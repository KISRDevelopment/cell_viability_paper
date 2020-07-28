import sys
import numpy as np
import numpy.random as rng
import pandas as pd
import os

def main(path, reps, folds, valid_prop):

    df = pd.read_csv(path)
    all_genes = list(set(df['a']).union(set(df['b'])))
    print("Total # genes: %d" % len(all_genes))

    chunk_size = len(all_genes) // folds 

    num_all_train = chunk_size * (folds - 1)
    num_test = len(all_genes) - num_all_train
    
    num_train = int(num_all_train * (1-valid_prop))
    num_valid = num_all_train - num_train

    n_genes = len(all_genes)

    print("# genes: %d, train: %d, valid: %d, test: %d" % (len(all_genes), num_train , num_valid, num_test))
    
    n = df.shape[0]
    train_sets = np.zeros((reps, folds, n), dtype=bool)
    valid_sets = np.zeros((reps, folds, n), dtype=bool)
    test_sets = np.zeros((reps, folds, n), dtype=bool)
    
    for r in range(reps):

        rng.shuffle(all_genes)

        for f in range(folds):
            
            from_idx = f * chunk_size
            to_idx = from_idx + chunk_size
            
            test_genes = all_genes[from_idx:to_idx]
            #assert len(test_genes) == num_test, 'Number of test=%d, expecting=%d' % (len(test_genes), num_test)

            all_train_genes = all_genes[:from_idx] + all_genes[to_idx:]
            #assert len(all_train_genes) == num_all_train, 'Number of all train=%d, expecting=%d' % (len(all_train_genes), num_all_train)

            rng.shuffle(all_train_genes)

            train_genes = all_train_genes[:num_train]
            valid_genes = all_train_genes[num_train:]
            print("train, valid, test genes: %d, %d, %d" % (len(train_genes), len(valid_genes), len(test_genes)))

            assert(len(set(train_genes).intersection(test_genes)) == 0)
            assert(len(set(train_genes).intersection(valid_genes)) == 0)
            assert(len(set(valid_genes).intersection(test_genes)) == 0)
            
            train_ix = df['a'].isin(train_genes) & df['b'].isin(train_genes)
            valid_ix = df['a'].isin(valid_genes) & df['b'].isin(valid_genes)
            test_ix = df['a'].isin(test_genes) & df['b'].isin(test_genes)

            sizes = np.array([np.sum(train_ix), np.sum(valid_ix), np.sum(test_ix)])
            print("Set sizes: %s (total = %d)" % (sizes, np.sum(sizes)))
            

            train_sets[r, f, :] = train_ix
            valid_sets[r, f, :] = valid_ix
            test_sets[r, f, :] = test_ix
    
    # ensure no repeated splits
    # unique_rows = np.unique(train_sets, axis=0)
    # assert(unique_rows.shape[0] == train_sets.shape[0])
    # unique_rows = np.unique(valid_sets, axis=0)
    # assert unique_rows.shape[0] == valid_sets.shape[0], '%d != %d' % (unique_rows.shape[0], valid_sets.shape[0])
    # unique_rows = np.unique(test_sets, axis=0)
    # assert(unique_rows.shape[0] == test_sets.shape[0])
    
    output_path = '../generated-data/splits/%s_%dreps_%dfolds_%0.2fvalid' % (os.path.basename(path), reps, folds, valid_prop)

    print("Writing to %s" % output_path)
    np.savez(output_path, 
        train_sets=train_sets, 
        valid_sets=valid_sets, 
        test_sets=test_sets,
        reps=reps,
        folds=folds,
        valid_prop=valid_prop)
    
if __name__ == "__main__":

    path = sys.argv[1]
    reps = int(sys.argv[2])
    folds = int(sys.argv[3])
    valid_prop = float(sys.argv[4])
    

    main(path, reps, folds, valid_prop)