import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np
import json 
import pandas as pd 
import sklearn.metrics 
import models.common 

class SingleInputNNModel:

    def __init__(self, model_spec):
        self._model_spec = model_spec

    def train(self, train_df, valid_df):

        model_spec = self._model_spec

        add_extra_info_to_spec(model_spec, train_df)
        model_spec['n_output_dim'] = models.common.calculate_output_dim(train_df, model_spec['target_col'])

        self._create_model()

        train_inputs = models.common.create_inputs(model_spec, train_df)
        train_inputs, mus, stds = models.common.normalize_inputs(train_inputs)
        train_Y = keras.utils.to_categorical(train_df[model_spec['target_col']])
        
        valid_inputs = models.common.create_inputs(model_spec, valid_df)
        valid_inputs, _, _ = models.common.normalize_inputs(valid_inputs, mus, stds)
        valid_Y = keras.utils.to_categorical(valid_df[model_spec['target_col']])

        earlystopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                    patience=model_spec['patience'], restore_best_weights=True)
        callbacks = [earlystopping_callback]

        self._model.fit(train_inputs, 
                train_Y, 
                batch_size=int(model_spec['batch_size_p'] * train_Y.shape[0]),
                epochs=model_spec['epochs'],
                verbose=model_spec['verbose'],
                validation_data=(valid_inputs, valid_Y),
                validation_batch_size=100000,
                callbacks=callbacks)

        self._mus = mus 
        self._stds = stds 

    def save(self, path):
        weights = self._model.get_weights()
        np.savez(path,
            model_spec=self._model_spec, 
            weights=np.array(weights, dtype=object), 
            mus=np.array(self._mus, dtype=object), 
            stds=np.array(self._stds, dtype=object))
    
    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        model_spec = d['model_spec'].item()
        weights = d['weights']
        mus = d['mus'].tolist()
        stds = d['stds'].tolist()
        
        m = SingleInputNNModel(model_spec)
        m._create_model()
        m._model.set_weights(weights)
        m._mus = mus 
        m._stds = stds 

        return m
    
    def predict(self, test_df):
        test_inputs = models.common.create_inputs(model_spec, test_df)
        test_inputs, _, _ = models.common.normalize_inputs(test_inputs, self._mus, self._stds)
        preds = self._model.predict(test_inputs, batch_size=1000000)
        return preds 

    def _create_model(self):

        model_spec = self._model_spec

        output_dim = model_spec['n_output_dim']

        input_layers = [keras.layers.Input(shape=(model_spec['feature_sets'][fs]['dim'],)) for 
            fs in model_spec['selected_feature_sets']]
        
        emb_layer = create_single_gene_embedding_module(model_spec)(input_layers)
        output_layer = keras.layers.Dense(output_dim, activation='softmax')(emb_layer)

        model = keras.Model(inputs=input_layers, outputs=output_layer)
        opt = keras.optimizers.Nadam(learning_rate=model_spec['learning_rate'])
        model.compile(opt, loss=models.common.weighted_categorical_xentropy)
        #print(model.summary())

        self._model = model

def create_single_gene_embedding_module(model_spec):
    selected_feature_sets = model_spec['selected_feature_sets']
    
    modules = {}

    input_layers = []
    module_outputs = []
    for feature_set in selected_feature_sets:
        feature_spec = model_spec['feature_sets'][feature_set]
        
        dim = feature_spec['dim']

        # get the appropriate module or create it if it hasn't been
        # created yet
        target_module_name = feature_spec['module']
        if target_module_name in modules:
            target_module = modules[target_module_name]
        else:
            module_spec = model_spec['modules'][target_module_name]
            target_module = create_module(module_spec, dim)
            modules[target_module_name] = target_module

        input_layer = keras.layers.Input(shape=(dim,))
        input_layers.append(input_layer)

        module_output = target_module(input_layer)
        module_outputs.append(module_output)
        
    # concatenate
    concatenated_module_output = keras.layers.Concatenate()(module_outputs)

    # embed
    emb_layer = keras.layers.Dense(model_spec['embedding_size'], 
        activation=model_spec['embedding_activation'])(concatenated_module_output)

    return keras.Model(inputs=input_layers, outputs=emb_layer)

def create_module(module_spec, dim):

    module = keras.Sequential()

    input_dim = dim 
    for layer_size in module_spec['layer_sizes']:
        module.add(keras.layers.Dense(layer_size, 
            input_shape=(input_dim,), 
            activation=module_spec['hidden_activation']))
        input_dim = layer_size 
    
    return module 

def add_extra_info_to_spec(model_spec, df):
    
    if 'selected_feature_sets' not in model_spec:
        model_spec['selected_feature_sets'] = list(model_spec['feature_sets'].keys())
        
    # add the feature dimensions and column names for each feature set
    for feature_set, props in model_spec['feature_sets'].items():
        ix = df.columns.str.startswith('%s-' % feature_set)
        props['dim'] = np.sum(ix)
        props['cols'] = list(df.columns[ix])


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

    splits = np.load(splits_path)['splits']
    split = splits[split]

    m = SingleInputNNModel(model_spec)
    
    train_df, valid_df, test_df = models.common.get_dfs(df, split)

    m.train(train_df, valid_df)
    m.save(model_output_path)
    
    m = SingleInputNNModel.load(model_output_path)

    preds = m.predict(test_df)

    models.common.evaluate(np.array(test_df['bin']), preds)

