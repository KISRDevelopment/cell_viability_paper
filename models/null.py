import numpy as np
import pandas as pd 
import models.common 
import tensorflow.keras as keras 

class NullModel:

    def __init__(self, model_spec, **kwargs):
        self._target_col = model_spec['target_col']

    def train(self, train_df, valid_df):
        
        train_Y = keras.utils.to_categorical(train_df[self._target_col])
        valid_Y = keras.utils.to_categorical(valid_df[self._target_col])
        Y = np.vstack((train_Y, valid_Y))

        self._mu = np.mean(Y, axis=0)

    
    def save(self, path):
        np.savez(path, mu=self._mu, target_col=self._target_col)
    
    @staticmethod
    def load(path, **kwargs):
        d = np.load(path, allow_pickle=True)
        m = NullModel({ 'target_col' : d['target_col'] })
        m._mu = d['mu'] 
        
        return m
    
    def predict(self, test_df, training_norm=True):
        preds = np.tile(self._mu, (test_df.shape[0], 1))
        return preds 

if __name__ == "__main__":
    import sys 

    dataset_path = sys.argv[1]
    splits_path = sys.argv[2]
    split = int(sys.argv[3])
    df = pd.read_feather(dataset_path)

    splits = np.load(splits_path, allow_pickle=True)['splits']
    split = splits[split]
    
    m = NullModel({ 'target_col' : 'bin' })
    
    train_df, valid_df, test_df = models.common.get_dfs(df, split)

    m.train(train_df, valid_df)
    m.save('../tmp/null.npz')
    
    m = NullModel.load('../tmp/null.npz')

    preds = m.predict(test_df)

    r = models.common.evaluate(np.array(test_df['bin']), preds)
    print(r)