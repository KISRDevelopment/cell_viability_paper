import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np
import json 
import pandas as pd 
import sklearn.metrics 
import models.common
import models.nn_single 

class TripleInputNNModel:

    def __init__(self, model_spec, sg_path):
        self._model_spec = model_spec
        self._sg_path = sg_path
        self._sgdf = pd.read_feather(sg_path) 

        sgs = model_spec['single_gene_spec']
        models.nn_single.add_extra_info_to_spec(sgs, self._sgdf)
        self._sg_inputs = create_sg_inputs(sgs, self._sgdf)

    def train(self, train_df, valid_df):
        model_spec = self._model_spec 
        dgs = model_spec['double_gene_spec']

        models.nn_single.add_extra_info_to_spec(dgs, train_df)
        model_spec['n_output_dim'] = models.common.calculate_output_dim(train_df, model_spec['target_col'])

        self._create_model() 

        # prepare pairwise inputs
        train_double_gene_inputs_ab = models.common.create_inputs(dgs, train_df, 'ab-')
        train_double_gene_inputs_ac = models.common.create_inputs(dgs, train_df, 'ac-')
        train_double_gene_inputs_bc = models.common.create_inputs(dgs, train_df, 'bc-')
        train_double_gene_inputs_all = train_double_gene_inputs_ab + train_double_gene_inputs_ac + train_double_gene_inputs_bc
        train_double_gene_inputs_all, mus, stds = models.common.normalize_inputs(train_double_gene_inputs_all)

        valid_double_gene_inputs_ab = models.common.create_inputs(dgs, valid_df, 'ab-')
        valid_double_gene_inputs_ac = models.common.create_inputs(dgs, valid_df, 'ac-')
        valid_double_gene_inputs_bc = models.common.create_inputs(dgs, valid_df, 'bc-')
        valid_double_gene_inputs_all = valid_double_gene_inputs_ab + valid_double_gene_inputs_ac + valid_double_gene_inputs_bc
        valid_double_gene_inputs_all, mus, stds = models.common.normalize_inputs(valid_double_gene_inputs_all)
        
        # outputs
        train_Y = keras.utils.to_categorical(train_df[model_spec['target_col']])
        valid_Y = keras.utils.to_categorical(valid_df[model_spec['target_col']])

        earlystopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                    patience=model_spec['patience'], restore_best_weights=True)
        callbacks = [earlystopping_callback]

        # create data iterators (necessary because some feature sets are too large to put in ram)
        train_iterator = create_data_iterator(train_df, train_Y, self._sg_inputs, train_double_gene_inputs_all, model_spec['batch_size'])
        valid_iterator = create_data_iterator(valid_df, valid_Y, self._sg_inputs, valid_double_gene_inputs_all, model_spec['batch_size']*10, False)

        self._model.fit(x=train_iterator(),
                steps_per_epoch=np.ceil(train_df.shape[0] / model_spec['batch_size']),
                epochs=model_spec['epochs'],
                verbose=model_spec['verbose'],
                validation_data=valid_iterator(),
                validation_steps=np.ceil(valid_df.shape[0] / model_spec['batch_size']),
                callbacks=callbacks)

        self._mus = mus 
        self._stds = stds 

    def save(self, path):
        weights = self._model.get_weights()
        np.savez(path,
            model_spec=self._model_spec, 
            sg_path=self._sg_path,
            weights=np.array(weights, dtype=object), 
            mus=np.array(self._mus, dtype=object), 
            stds=np.array(self._stds, dtype=object)) 
    
    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        model_spec = d['model_spec'].item()
        sg_path = d['sg_path'].item()
        weights = d['weights']
        mus = d['mus'].tolist()
        stds = d['stds'].tolist()
        
        m = TripleInputNNModel(model_spec, sg_path)
        m._create_model()
        m._model.set_weights(weights)
        m._mus = mus 
        m._stds = stds 

        return m 
    
    def predict(self, test_df):

        dgs = self._model_spec['double_gene_spec']
        test_double_gene_inputs_ab = models.common.create_inputs(dgs, test_df, 'ab-')
        test_double_gene_inputs_ac = models.common.create_inputs(dgs, test_df, 'ac-')
        test_double_gene_inputs_bc = models.common.create_inputs(dgs, test_df, 'bc-')
        test_double_gene_inputs_all = test_double_gene_inputs_ab + test_double_gene_inputs_ac + test_double_gene_inputs_bc
        test_double_gene_inputs_all, _, _ = models.common.normalize_inputs(test_double_gene_inputs_all, self._mus, self._stds)

        batch_size =  self._model_spec['batch_size']*10
        test_iterator = create_data_iterator(test_df, np.zeros((test_df.shape[0], self._model_spec['n_output_dim'])),
            self._sg_inputs, test_double_gene_inputs_all, batch_size, False)

        preds = self._model.predict(x=test_iterator(), steps=np.ceil(test_df.shape[0] / batch_size))
        
        return preds 
    
    def _create_model(self):
        model_spec = self._model_spec

        output_dim = model_spec['n_output_dim']

        sgs = model_spec['single_gene_spec']
        dgs = model_spec['double_gene_spec']

        has_single_gene_features = len(sgs['selected_feature_sets']) > 0
        has_double_gene_features = len(dgs['selected_feature_sets']) > 0

        inputs_a = []
        inputs_b = []
        inputs_c = []
        inputs_ab = []
        inputs_ac = []
        inputs_bc = []
        if has_single_gene_features:
            single_gene_emb_module = models.nn_single.create_single_gene_embedding_module(sgs)
            inputs_a = [keras.layers.Input(shape=(sgs['feature_sets'][fs]['dim'],)) for fs in sgs['selected_feature_sets']]
            inputs_b = [keras.layers.Input(shape=(sgs['feature_sets'][fs]['dim'],)) for fs in sgs['selected_feature_sets']]
            inputs_c = [keras.layers.Input(shape=(sgs['feature_sets'][fs]['dim'],)) for fs in sgs['selected_feature_sets']]
            
            output_a = single_gene_emb_module(inputs_a)
            output_b = single_gene_emb_module(inputs_b)
            output_c = single_gene_emb_module(inputs_c)
        
        if has_double_gene_features:
            double_gene_emb_module = models.nn_single.create_single_gene_embedding_module(dgs)
            inputs_ab = [keras.layers.Input(shape=(dgs['feature_sets'][fs]['dim'],)) for fs in dgs['selected_feature_sets']]
            inputs_ac = [keras.layers.Input(shape=(dgs['feature_sets'][fs]['dim'],)) for fs in dgs['selected_feature_sets']]
            inputs_bc = [keras.layers.Input(shape=(dgs['feature_sets'][fs]['dim'],)) for fs in dgs['selected_feature_sets']]

            output_ab = double_gene_emb_module(inputs_ab)
            output_ac = double_gene_emb_module(inputs_ac)
            output_bc = double_gene_emb_module(inputs_bc)
        
        if has_single_gene_features and not has_double_gene_features:
            merged = (output_a + output_b + output_c) / 3
        elif has_single_gene_features and has_double_gene_features:
            merged = keras.layers.Concatenate()([(output_a + output_b + output_c)/3, (output_ab + output_ac + output_bc)/3])
        elif not has_single_gene_features and has_double_gene_features:
            merged = (output_ab + output_ac + output_bc)/3 
        else:
            raise Exception("Must have at least single or double features.")
        
        output_layer = keras.layers.Dense(output_dim, activation='softmax')(merged)

        model = keras.Model(inputs=inputs_a + inputs_b + inputs_c + inputs_ab + inputs_ac + inputs_bc, outputs=output_layer)
        opt = keras.optimizers.Nadam(learning_rate=model_spec['learning_rate'])
        model.compile(opt, loss=models.common.weighted_categorical_xentropy)
        #print(model.summary())

        self._model = model

def create_sg_inputs(model_spec, df):
    inputs = []
    for feature_set in model_spec['selected_feature_sets']:
        props = model_spec['feature_sets'][feature_set]
        F = np.array(df[ props['cols'] ])
        F,_,_ = models.common.normalize(F)
        fdf = pd.DataFrame(data=F, index=df['id'], columns=props['cols'])
        inputs.append(fdf)
    return inputs

def create_data_iterator(df, Y, single_fsets, pairwise_fsets, batch_size, training=True):
    idx = np.arange(df.shape[0])
    
    def iterator():
        while True:
            if training:
                np.random.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                indecies = idx[i:(i+batch_size)]
                
                batch_df = df.iloc[indecies]
                batch_Y = Y[indecies,:]

                a_id = batch_df['a_id']
                b_id = batch_df['b_id']
                c_id = batch_df['c_id']

                inputs_a = [np.array(fs.loc[a_id]) for fs in single_fsets]
                inputs_b = [np.array(fs.loc[b_id]) for fs in single_fsets]
                inputs_c = [np.array(fs.loc[c_id]) for fs in single_fsets]

                inputs_all_pairs = [fs[indecies,:] for fs in pairwise_fsets] 

                inputs = inputs_a + inputs_b + inputs_c + inputs_all_pairs
                
                yield (inputs, batch_Y)
    
    return iterator 

if __name__ == "__main__":
    import sys 

    model_spec_path = sys.argv[1]
    smf_dataset_path = sys.argv[2]
    dataset_path = sys.argv[3]
    splits_path = sys.argv[4]
    split = int(sys.argv[5])
    model_output_path = sys.argv[6]

    with open(model_spec_path, 'r') as f:
        model_spec = json.load(f)
    
    df = pd.read_feather(dataset_path)

    splits = np.load(splits_path, allow_pickle=True)['splits']
    split = splits[split]

    m = TripleInputNNModel(model_spec, smf_dataset_path)
    
    train_df, valid_df, test_df = models.common.get_dfs(df, split)

    m.train(train_df, valid_df)

    m.save(model_output_path)

    m = TripleInputNNModel.load(model_output_path)


    preds = m.predict(test_df)

    models.common.evaluate(np.array(test_df[model_spec['target_col']]), preds)
