import numpy as np 
import pandas as pd 
import sklearn.model_selection
import keras.utils 
import numpy.random as rng 

def main(train_path, test_path, p_valid, combined_df_path, output_path):

    train_df = pd.read_csv(train_path)
    
    train_genes = get_genes(train_df)
    rng.shuffle(train_genes)

    n_valid = int(p_valid * len(train_genes))
    valid_genes = train_genes[:n_valid]
    train_genes = train_genes[n_valid:]
    print("Training genes: %d, Valid genes: %d" % (len(train_genes), len(valid_genes)))

    train_ix = train_df['a'].isin(train_genes) & train_df['b'].isin(train_genes)
    valid_ix = train_df['a'].isin(valid_genes) & train_df['b'].isin(valid_genes)
    
    valid_df = train_df[valid_ix].copy()
    valid_df['split'] = 'valid'

    train_df = train_df[train_ix].copy()
    train_df['split'] = 'train'

    test_df = pd.read_csv(test_path)
    test_df['split'] = 'test'
    test_genes = get_genes(test_df)

    combined_df = pd.concat((train_df, valid_df, test_df), axis=0)
    print("Total size: %d" % combined_df.shape[0])

    n = combined_df.shape[0]

    assert set(train_genes).intersection(valid_genes) == set()
    assert set(train_genes).intersection(test_genes) == set()
    assert set(valid_genes).intersection(test_genes) == set()

    train_sets = np.zeros((1, 1, n), dtype=bool)
    valid_sets = np.zeros((1, 1, n), dtype=bool)
    test_sets = np.zeros((1, 1, n), dtype=bool)
    
    train_sets[0,0,:] = combined_df['split'] == 'train'
    valid_sets[0,0,:] = combined_df['split'] == 'valid'
    test_sets[0,0,:] = combined_df['split'] == 'test'

    print(np.sum(train_sets))
    print(np.sum(valid_sets))
    print(np.sum(test_sets))

    print("Writing to %s" % output_path)
    np.savez(output_path, 
        train_sets=train_sets, 
        valid_sets=valid_sets, 
        test_sets=test_sets,
        valid_prop=p_valid)

    combined_df.to_csv(combined_df_path, index=False)

def smf(train_path, test_path, p_valid, combined_df_path, output_path):
    train_df = pd.read_csv(train_path)

    ix = rng.permutation(train_df.shape[0])
    
    n_valid = int(p_valid * train_df.shape[0])
    valid_ix = ix[:n_valid]

    split = np.array(['train' for i in range(train_df.shape[0])])
    split[valid_ix] = 'valid'
    train_df['split'] = split 

    test_df = pd.read_csv(test_path)
    test_df['split'] = 'test'

    combined_df = pd.concat((train_df, test_df), axis=0)
    n = combined_df.shape[0]

    # populate master list of all CV test indecies
    train_sets = np.zeros((1, 1, n), dtype=bool)
    valid_sets = np.zeros((1, 1, n), dtype=bool)
    test_sets = np.zeros((1, 1, n), dtype=bool)
    
    train_sets[0,0,:] = combined_df['split'] == 'train'
    valid_sets[0,0,:] = combined_df['split'] == 'valid'
    test_sets[0,0,:] = combined_df['split'] == 'test'


    print(np.sum(train_sets))
    print(np.sum(valid_sets))
    print(np.sum(test_sets))

    print("Writing to %s" % output_path)
    np.savez(output_path, 
        train_sets=train_sets, 
        valid_sets=valid_sets, 
        test_sets=test_sets,
        valid_prop=p_valid)

    combined_df.to_csv(combined_df_path, index=False)

def get_genes(df):

    return list(set(df['a']).union(set(df['b'])))
if __name__ == "__main__":
    import sys 
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4])