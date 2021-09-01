import numpy as np 
import pandas as pd 
import sklearn.model_selection
import keras.utils 
import numpy.random as rng 

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

def gi(task_path, test_prop, train_output_path, test_output_path):

    df = pd.read_csv(task_path)
    all_genes = list(set(df['a']).union(set(df['b'])))
    print("Total # genes: %d" % len(all_genes))
    rng.shuffle(all_genes)


    bins = np.array(df['bin'])
    bins = keras.utils.to_categorical(bins)

    n_train = int(len(all_genes) * (1-test_prop))

    train_genes = all_genes[:n_train]
    test_genes = all_genes[n_train:]
    
    train_ix = df['a'].isin(train_genes) & df['b'].isin(train_genes)
    test_ix = df['a'].isin(test_genes) & df['b'].isin(test_genes)

    print("Train props:")
    train_df = df[train_ix]
    print(np.array([np.sum(train_df['bin'] == b) for b in [0,1,2,3]]) / train_df.shape[0])
    print("Train size: %d" % train_df.shape[0])
    train_df.to_csv(train_output_path, index=False)

    print("Test props:")
    test_df = df[test_ix]
    print(np.array([np.sum(test_df['bin'] == b) for b in [0,1,2,3]]) / test_df.shape[0])
    print("Test size: %d" % test_df.shape[0])
    test_df.to_csv(test_output_path, index=False)

def tgi(task_path, test_prop, train_output_path, test_output_path):

    df = pd.read_csv(task_path)
    all_genes = list(set(df['a']).union(set(df['b'])))
    print("Total # genes: %d" % len(all_genes))
    rng.shuffle(all_genes)


    bins = np.array(df['bin'])
    bins = keras.utils.to_categorical(bins)

    n_train = int(len(all_genes) * (1-test_prop))

    train_genes = all_genes[:n_train]
    test_genes = all_genes[n_train:]
    
    train_ix = df['a'].isin(train_genes) & df['b'].isin(train_genes) & df['c'].isin(train_genes)
    test_ix = df['a'].isin(test_genes) & df['b'].isin(test_genes) & df['c'].isin(test_genes)

    print("Train props:")
    train_df = df[train_ix]
    print(np.array([np.sum(train_df['bin'] == b) for b in [0,1,2,3]]) / train_df.shape[0])
    print("Train size: %d" % train_df.shape[0])
    #train_df.to_csv(train_output_path, index=False)

    print("Test props:")
    test_df = df[test_ix]
    print(np.array([np.sum(test_df['bin'] == b) for b in [0,1,2,3]]) / test_df.shape[0])
    print("Test size: %d" % test_df.shape[0])
    
    #test_df.to_csv(test_output_path, index=False)

    return (train_df.shape[0], test_df.shape[0])

if __name__ == "__main__":
    import sys 
    task_path = sys.argv[1]
    #gi(task_path,0.2,"","")
    #main(task_path, 0.2, "", "")

    n_trains = []
    n_tests = []
    for i in range(50):
        n_train, n_test = tgi(task_path, 0.5, "", "")
        n_trains.append(n_train)
        n_tests.append(n_test)

    print(n_trains)

    print("Average train size: %f, test: %f" % (np.mean(n_trains), np.mean(n_tests)))