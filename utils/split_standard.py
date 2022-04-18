import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

SEED = 425345
rng = np.random.RandomState(SEED)
def main(dataset_path, reps, folds, valid_p, dev_test, test_p, output_path, **kwargs):

    smf_df = pd.read_csv(dataset_path)
    y = np.array(smf_df['bin'])

    # split data into development and testing
    if dev_test:
        test_n = int(1/test_p)
        is_dev = split_train_test_stratified(test_n, y)
        dev_indecies = np.where(is_dev)[0]
    else:
        dev_indecies = np.arange(smf_df.shape[0])
    
    print("# samples for CV: %d" % len(dev_indecies))
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=reps, random_state=rng)

    # store split indecies
    splits = np.zeros((reps * folds, smf_df.shape[0]))
    i = 0
    for train_index, test_index in rskf.split(np.zeros(len(dev_indecies)), y[dev_indecies]):
        
        train_indecies = dev_indecies[train_index]
        train_y = y[train_indecies]
        
        # create train/validation out of training set
        valid_n = int(1/valid_p)
        is_train = split_train_test_stratified(valid_n, train_y)
        
        valid_indecies = train_indecies[~is_train]
        train_indecies = train_indecies[is_train]
        test_indecies = dev_indecies[test_index]
        
        # create split
        split = np.zeros(smf_df.shape[0])
        split[train_indecies] = 1
        split[valid_indecies] = 2
        split[test_indecies] = 3
        
        # ensure no overlap
        assert set(train_indecies) & set(test_indecies) & set(valid_indecies) == set()
        
        # ensure they match dev indecies
        assert set(train_indecies) | set(valid_indecies) | set(test_indecies) == set(dev_indecies)
        
        splits[i, :] = split
        
        rep, fold = (i // folds), (i % folds)
        # print_props("[%d, %d] Train props:" % (rep,fold), smf_df, split == 1)
        # print_props("[%d, %d] Valid props:"% (rep,fold), smf_df, split == 2)
        # print_props("[%d, %d] Test props:"% (rep,fold), smf_df, split == 3)
        # print()

        i+=1

    uni = np.unique(splits, axis=0)
    assert uni.shape[0] == (reps*folds)

    if dev_test:
        assert np.all(np.sum(splits, axis=0)[~is_dev] == 0)
    
    np.savez(output_path, splits=splits, 
                          reps=reps, 
                          folds=folds,
                          valid_p=valid_p,
                          dev_test=dev_test,
                          test_p=test_p)

def split_train_test_stratified(n_splits, y):
    """ Just do a single stratified training and testing split """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
    is_train = np.zeros(y.shape[0]).astype(bool)
    for train_index, test_index in skf.split(np.zeros(y.shape[0]), y):
        is_train[train_index] = 1
        break
    return is_train

def print_props(title, df, ix):
    counts = np.array([np.sum(df[ix]['bin'] == b) for b in sorted(set(df[ix]['bin']))])
    print("%32s" % title, end=' ')
    print(counts / np.sum(counts), end=' ')
    print("Total: %d" % np.sum(counts))
if __name__ == "__main__":
    import sys 
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), False, 5, sys.argv[4])