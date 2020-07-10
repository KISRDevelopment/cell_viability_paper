import numpy as np
import numpy.random as rng
import sys 
import pandas as pd 
import numpy as np 
import os 

def main():
    path = sys.argv[1]
    reps = int(sys.argv[2])
    folds = int(sys.argv[3])
    valid_prop = float(sys.argv[4])

    train_sets, valid_sets, test_sets = generate_cv_splits(path, reps, folds, valid_prop)

    output_path = '../generated-data/splits/%s_%dreps_%dfolds_%0.2fvalid' % (os.path.basename(path), reps, folds, valid_prop)
    print("Writing to %s" % output_path)
    np.savez(output_path, 
        train_sets=train_sets,
        valid_sets=valid_sets,
        test_sets=test_sets,
        reps=reps,
        folds=folds,
        valid_prop=valid_prop)

def generate_cv_splits(path, reps, folds, valid_prop):
    
    df = pd.read_csv(path)

    n = df.shape[0]
    
    print("dataset size: %d" % n)
    
    # create splitter instance
    splitter = StandardCvSplitter()
    
    # populate master list of all CV test indecies
    train_sets = np.zeros((reps, folds, n), dtype=bool)
    valid_sets = np.zeros((reps, folds, n), dtype=bool)
    test_sets = np.zeros((reps, folds, n), dtype=bool)
    
    for r in range(reps):
        for fid, tup in enumerate(splitter(n, folds)):
            train_ix, test_ix = tup
            test_sets[r, fid, :] = test_ix

            train_indecies = np.where(train_ix)[0]
            rng.shuffle(train_indecies)

            valid_count = int(len(train_indecies) * valid_prop)
            valid_indecies = train_indecies[:valid_count]
            train_indecies = train_indecies[valid_count:]

            train_sets[r, fid, train_indecies] = 1
            valid_sets[r, fid, valid_indecies] = 1

            # ensure no overlap between training and testing
            assert(np.sum(train_sets[r,fid,:] * valid_sets[r,fid,:]) == 0)

            # ensure validation count
            assert(np.sum(valid_sets[r,fid,:]) == valid_count)

            # ensure training count
            assert np.sum(train_sets[r,fid,:]) == len(train_indecies), "Len train: %d (expected %d)" % (np.sum(train_sets[r,fid,:]), len(train_indecies))
            
            # ensure total count
            assert(np.sum(train_sets[r,fid,:]) + np.sum(valid_sets[r,fid,:]) + np.sum(test_sets[r,fid,:]) == n)
            print([np.sum(train_sets[r,fid,:]) , np.sum(valid_sets[r,fid,:]) , np.sum(test_sets[r,fid,:])])

        # ensure each instance is tested exactly once
        curr_ix = np.zeros(n)
        for fid in range(folds):
            curr_ix += test_sets[r, fid, :]
        assert(np.sum(curr_ix) == n)
    
    # ensure no repeated splits
    f_train_sets = np.reshape(train_sets, (reps*folds, n))
    unique_rows = np.unique(f_train_sets, axis=0)
    assert(unique_rows.shape[0] == f_train_sets.shape[0])
    
    f_test_sets = np.reshape(test_sets, (reps*folds,n))
    unique_rows = np.unique(f_test_sets, axis=0)
    assert(unique_rows.shape[0] == f_test_sets.shape[0])
    
    f_valid_sets = np.reshape(valid_sets, (reps*folds,n))
    unique_rows = np.unique(f_valid_sets, axis=0)
    assert(unique_rows.shape[0] == f_valid_sets.shape[0])
    
    return train_sets, valid_sets, test_sets

class StandardCvSplitter(object):
    
    def __init__(self, df=None):
        pass
    
    def __call__(self, n, folds):
        
        chunk_size = -(-n // folds)
    
        indecies = list(range(n))
        rng.shuffle(indecies)
        
        for i in range(0, n, chunk_size):
            test_indecies = indecies[i:i+chunk_size]
            
            test_ix = np.zeros(n, dtype=bool)
            test_ix[test_indecies] = 1
            
            # we retain both train and test ix even though one is 
            # complement of the other because for other splitters,
            # this is not always the case (e.g., NodeStratifiedCvSplitter)
            yield (1-test_ix).astype(bool), test_ix
            

if __name__ == "__main__":
    main()
    