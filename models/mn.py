import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np
import json 
import pandas as pd 
import sklearn.metrics 
import models.common 

class MnModel:

    def __init__(self, model_spec):
        self._model_spec = model_spec 
    
    def train(self, train_df, valid_df):
        model_spec = self._model_spec

        self._add_extra_info_to_spec(train_df)

        model = self._create_model()

        train_X = np.array(train_df[model_spec['features']])
        train_X, mu, std = models.common.normalize(train_X)
        train_Y = keras.utils.to_categorical(train_df[model_spec['target_col']])
        
        valid_X = np.array(valid_df[model_spec['features']])
        valid_X, _, _ = models.common.normalize(valid_X, mu, std)
        valid_Y = keras.utils.to_categorical(valid_df[model_spec['target_col']])

        earlystopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                    patience=model_spec['patience'], restore_best_weights=True)
        callbacks = [earlystopping_callback]

        model.fit(train_X, 
                train_Y, 
                batch_size=int(model_spec['batch_size_p'] * train_Y.shape[0]),
                epochs=model_spec['epochs'],
                verbose=model_spec['verbose'],
                validation_data=(valid_X, valid_Y),
                validation_batch_size=100000,
                callbacks=callbacks)

        self._model = model 
        self._mu = mu 
        self._std = std 
    
    def save(self, path):
        weights = self._model.get_weights()
        np.savez(path,
            model_spec=self._model_spec, 
            weights=np.array(weights, dtype=object), 
            mu=self._mu, 
            std=self._std)
    
    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        model_spec = d['model_spec'].item()
        weights = d['weights']
        mu = d['mu']
        std = d['std']
        
        m = MnModel(model_spec)
        m._model = m._create_model()
        m._model.set_weights(weights)
        m._mu = mu 
        m._std = std 

        return m
    
    def predict(self, test_df):
        test_X = np.array(test_df[model_spec['features']])
        test_X, _, _ = models.common.normalize(test_X, self._mu, self._std)
        preds = self._model.predict(test_X, batch_size=1000000)
        return preds 
    
    def _add_extra_info_to_spec(self, df):
        model_spec = self._model_spec

        # calculate output dimension size (number of classes)
        model_spec['n_output_dim'] = np.unique(df[model_spec['target_col']]).shape[0]

        # calculate actual features
        ix = np.zeros_like(df.columns, dtype=bool)
        for f in model_spec['features']:
            ix = ix | df.columns.str.startswith(f)
        model_spec['features'] = list(df.columns[ix])

    def _create_model(self):
        
        model_spec = self._model_spec

        input_layer = keras.layers.Input(shape=(len(model_spec['features'],)))
        output_layer = keras.layers.Dense(model_spec['n_output_dim'], activation='softmax')(input_layer)

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        opt = keras.optimizers.Nadam(learning_rate=model_spec['learning_rate'])
        model.compile(opt, loss=models.common.weighted_categorical_xentropy)
        #print(model.summary())

        return model

if __name__ == "__main__":
    import sys 

    model_spec_path = sys.argv[1]
    dataset_path = sys.argv[2]
    splits_path = sys.argv[3]
    split = int(sys.argv[4])
    model_output_path = sys.argv[5]

    with open(model_spec_path, 'r') as f:
        model_spec = json.load(f)
    
    model_spec['epochs'] = 1000

    df = pd.read_feather(dataset_path)

    splits = np.load(splits_path, allow_pickle=True)['splits']
    split = splits[split]
    
    m = MnModel(model_spec)
    
    train_df, valid_df, test_df = models.common.get_dfs(df, split)

    m.train(train_df, valid_df)
    m.save(model_output_path)
    
    m = MnModel.load(model_output_path)

    preds = m.predict(test_df)

    models.common.evaluate(np.array(test_df['bin']), preds)
