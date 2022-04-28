import os 
import sklearn.model_selection
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold

def main():


    cfgs = [

        {
            "dataset_path" : "../generated-data/dataset_yeast_smf.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 5,
            "valid_p" : 0.2,
            "dev_test" : True,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_yeast_smf_dev_test"
        },
        {
            "dataset_path" : "../generated-data/dataset_yeast_smf.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 5,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_yeast_smf"
        },
        {
            "dataset_path" : "../generated-data/dataset_pombe_smf.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 5,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_pombe_smf"
        },
        {
            "dataset_path" : "../generated-data/dataset_human_smf.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 5,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_human_smf"
        },
        {
            "dataset_path" : "../generated-data/dataset_human_smf_ca_mo_v.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 5,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_human_smf_ca_mo_v"
        },
        {
            "dataset_path" : "../generated-data/dataset_human_smf_mo_v.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 5,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_human_smf_mo_v"
        },
        {
            "dataset_path" : "../generated-data/dataset_dro_smf.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 5,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_dro_smf"
        },
        {
            "dataset_path" : "../generated-data/dataset_dro_smf_ca_mo_v.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 5,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_dro_smf_ca_mo_v"
        },
        {
            "dataset_path" : "../generated-data/dataset_dro_smf_mo_v.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 5,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_dro_smf_mo_v"
        },
        {
            "dataset_path" : "../generated-data/dataset_yeast_gi_costanzo.feather",
            "function" : gi_split,
            "reps" : 10,
            "folds" : 4,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_yeast_gi_costanzo"
        },
        {
            "dataset_path" : "../generated-data/dataset_yeast_gi_hybrid.feather",
            "function" : gi_split,
            "reps" : 10,
            "folds" : 4,
            "valid_p" : 0.2,
            "dev_test" : True,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_yeast_gi_hybrid_dev_test"
        },
        {
            "dataset_path" : "../generated-data/dataset_yeast_gi_hybrid.feather",
            "function" : gi_split,
            "reps" : 10,
            "folds" : 4,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_yeast_gi_hybrid"
        },
        {
            "dataset_path" : "../generated-data/dataset_pombe_gi.feather",
            "function" : gi_split,
            "reps" : 10,
            "folds" : 4,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_pombe_gi"
        },
        {
            "dataset_path" : "../generated-data/dataset_human_gi.feather",
            "function" : gi_split,
            "reps" : 10,
            "folds" : 4,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_human_gi"
        },
        {
            "dataset_path" : "../generated-data/dataset_dro_gi.feather",
            "function" : gi_split,
            "reps" : 10,
            "folds" : 4,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_dro_gi"
        },
        {
            "dataset_path" : "../generated-data/dataset_yeast_tgi.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 4,
            "valid_p" : 0.2,
            "dev_test" : True,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_yeast_tgi_dev_test"
        },
        {
            "dataset_path" : "../generated-data/dataset_yeast_tgi.feather",
            "function" : standard_split,
            "reps" : 10,
            "folds" : 4,
            "valid_p" : 0.2,
            "dev_test" : False,
            "test_p" : 0.2,
            "output_path" : "../generated-data/splits/dataset_yeast_tgi"
        },
    ]

    os.makedirs("../generated-data/splits", exist_ok=True)
    for cfg in cfgs:
        print(cfg['dataset_path'], '->', cfg['output_path'])
        cfg['function'](**cfg)



def standard_split(dataset_path, reps, folds, valid_p, dev_test, test_p, output_path, **kwargs):

    SEED = 425345
    rng = np.random.RandomState(SEED)
    smf_df = pd.read_feather(dataset_path)
    y = np.array(smf_df['bin'])

    # split data into development and testing
    if dev_test:
        test_n = int(1/test_p)
        is_dev = split_train_test_stratified(test_n, y, rng)
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
        is_train = split_train_test_stratified(valid_n, train_y, rng)
        
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

def split_train_test_stratified(n_splits, y, rng):
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

def gi_split(dataset_path, reps, folds, valid_p, dev_test, test_p, output_path=None, **kwargs):
    SEED = 425345
    rng = np.random.RandomState(SEED)

    df = pd.read_feather(dataset_path)
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
    
    rkf = RepeatedKFold(n_splits=folds, 
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
    main()
