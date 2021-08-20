import numpy as np
import numpy.random as rng
import sys 
import pandas as pd 
import numpy as np 
import os 
import keras.utils 

def main(path, reps, folds, valid_prop, add_output_path_tag=False):
    
    train_sets, valid_sets, test_sets = generate_cv_splits(path, reps, folds, valid_prop)

    tag = '_simple' if add_output_path_tag else ''
    output_path = '../generated-data/splits/%s_%dreps_%dfolds_%0.2fvalid%s' % (os.path.basename(path), reps, folds, valid_prop, tag)
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
    bins = keras.utils.to_categorical(df['bin'])

    n = df.shape[0]
    
    print("dataset size: %d" % n)
    print("Overall props:")
    print(np.sum(bins, axis=0) / np.sum(bins))
    
    # create splitter instance
    splitter = StandardCvSplitter()
    
    # populate master list of all CV test indecies
    train_sets = np.zeros((reps, folds, n), dtype=bool)
    valid_sets = np.zeros((reps, folds, n), dtype=bool)
    test_sets = np.zeros((reps, folds, n), dtype=bool)
    
    for r in range(reps):
        for fid, tup in enumerate(splitter(bins, folds)):
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
            #print([np.sum(train_sets[r,fid,:]) , np.sum(valid_sets[r,fid,:]) , np.sum(test_sets[r,fid,:])])

            # print test class distribution
            test_df = df[test_ix]
            print("Set sizes: Train=%d, Valid=%d, Test=%d" % (len(train_indecies), len(valid_indecies), np.sum(test_ix)))
            print([np.sum(test_df['bin'] == b) / test_df.shape[0] for b in sorted(set(test_df['bin']))])
            
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
    
    def __call__(self, bins, folds):
        
        n = bins.shape[0]

        per_class_indecies = []
        per_class_chunk_sizes = []
        for c in range(bins.shape[1]):
            class_indecies = np.where(bins[:,c] == 1)[0]
            rng.shuffle(class_indecies)
            per_class_indecies.append(class_indecies)
            
            chunk_size = -(-len(class_indecies) // folds)
            per_class_chunk_sizes.append(chunk_size)

        for i in range(folds):

            test_indecies = []
            for c in range(bins.shape[1]):
                chunk_size = per_class_chunk_sizes[c]
                start = i * chunk_size
                end = start + chunk_size
                class_test_indecies = per_class_indecies[c][start:end]
                test_indecies.extend(class_test_indecies)
            
            test_ix = np.zeros(n, dtype=bool)
            test_ix[test_indecies] = 1
            
            # we retain both train and test ix even though one is 
            # complement of the other because for other splitters,
            # this is not always the case (e.g., NodeStratifiedCvSplitter)
            yield (1-test_ix).astype(bool), test_ix
            

if __name__ == "__main__":

    path = sys.argv[1]
    reps = int(sys.argv[2])
    folds = int(sys.argv[3])
    valid_prop = float(sys.argv[4])

    main(path, reps, folds, valid_prop)
    