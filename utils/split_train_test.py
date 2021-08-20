import numpy as np 
import pandas as pd 
import sklearn.model_selection
import keras.utils 

def main(task_path, test_prop, train_output_path, test_output_path):

    df = pd.read_csv(task_path)
    
    bins = np.array(df['bin'])
    bins = keras.utils.to_categorical(bins)

    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_prop)

    print("Class props:")
    print(np.sum(bins, axis=0) / np.sum(bins))
    for train_ix, test_ix in sss.split(np.zeros(bins.shape[0]), bins):

        print(train_ix.shape)
        print(test_ix.shape)

        print("Train props:")
        print(np.sum(bins[train_ix, :], axis=0) / np.sum(bins[train_ix]))

        print("Test props:")
        print(np.sum(bins[test_ix, :], axis=0) / np.sum(bins[test_ix]))

    
    ix = np.zeros(df.shape[0])
    ix[train_ix] = 1

    df['is_train'] = ix 
    print(np.sum(df['is_train']))
    print(df)
    
    print("Train props:")
    train_df = df[df['is_train'] == 1]
    print(np.array([np.sum(train_df['bin'] == b) for b in [0,1,2]]) / train_df.shape[0])
    print("Train size: %d" % train_df.shape[0])
    train_df.to_csv(train_output_path, index=False)

    print("Test props:")
    test_df = df[df['is_train'] == 0]
    print(np.array([np.sum(test_df['bin'] == b) for b in [0,1,2]]) / test_df.shape[0])
    print("Test size: %d" % test_df.shape[0])
    test_df.to_csv(test_output_path, index=False)

    

if __name__ == "__main__":
    import sys 
    task_path = sys.argv[1]
    bins_path = sys.argv[2]
    test_prop = float(sys.argv[3])
    main(task_path, bins_path, test_prop)
