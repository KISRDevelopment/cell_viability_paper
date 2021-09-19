import pandas as pd
import numpy as np 

import sys 
import os
import json 

import tensorflow as tf
import keras.backend as K
import keras.layers as layers
import keras.initializers as kinit
import keras.regularizers as regularizers
import keras.constraints as constraints
import keras.models 
import keras.optimizers 
import sklearn.metrics
import scipy.stats as stats 
import numpy.random as rng
import utils.eval_funcs as eval_funcs
from keras.utils import np_utils

import scipy.sparse 
import models.feature_loader
from termcolor import colored

import tensorflowjs as tfjs 

def main(cfg, rep, fold, output_path, print_results=True, return_model=False):
    K.clear_session()

    dataset_path = cfg['task_path']
    train_test_path = cfg['splits_path']
    
    # load dataset
    df = pd.read_csv(dataset_path)
    
    # create output
    Y = np_utils.to_categorical(df[cfg['target_col']])
    
    # load train/test split 
    data = np.load(train_test_path)
    train_sets = data['train_sets']
    valid_sets = data['valid_sets']
    test_sets = data['test_sets']
    
    # create training and testing data frames
    train_ix = train_sets[rep, fold,:]
    valid_ix = valid_sets[rep,fold,:]
    test_ix = test_sets[rep,fold,:]
    
    
    print("Dataset size: %d, Total: %d" % (df.shape[0], np.sum(train_ix + valid_ix + test_ix)))
    
    if not cfg.get("early_stopping", True):
        train_ix = train_ix + valid_ix
        
    if cfg.get("train_on_full_dataset", False):
        print(colored("******** TRAINING ON FULL DATASET ***********", "red"))
        train_ix = np.ones_like(train_ix)
    
    if cfg.get("test_on_full_dataset", False):
        print(colored("******** TESTING ON FULL DATASET ***********", "green"))
        test_ix = np.ones_like(test_ix)
    

    train_df = df.iloc[train_ix]
    valid_df = df.iloc[valid_ix]
    test_df = df.iloc[test_ix]

    print("Train size: %d, valid: %d, test: %d" % (train_df.shape[0], valid_df.shape[0], test_df.shape[0]))
    
    train_Y = Y[train_ix,:]
    valid_Y = Y[valid_ix,:]
    test_Y = Y[test_ix, :]

    if cfg.get("bootstrap_training", False):
        print(colored("******** BOOTSTRAPPING TRAINING ***********", "blue"))
        rix = rng.choice(train_df.shape[0], train_df.shape[0], replace=True)
        train_df = train_df.iloc[rix]
        train_Y = train_Y[rix,:]
        
    fsets, feature_labels = load_features(cfg)

    #
    # NN definition
    #
    input_node = layers.Input(shape=(len(feature_labels),), name='input_features')
    output_layer = layers.Dense(Y.shape[1], activation='softmax')
    output_node = output_layer(input_node)

    if cfg['balanced_loss']:
        loss = weighted_categorical_xentropy
    else:
        loss = 'categorical_crossentropy'
    
    model = keras.models.Model(inputs=input_node, outputs=output_node)
    model.compile(cfg['optimizer'], loss)

    if cfg.get("trained_model_path", None) is not None:
        if cfg.get("add_repfold_to_trained_model_path", True):
            cfg["trained_model_path"] = "%s_%d_%d" % (cfg["trained_model_path"], rep, fold)
    

    # train
    if cfg.get("train_model", True):
        train_iterator = create_data_iterator(train_df, train_Y, fsets, cfg)
        valid_iterator = create_data_iterator(valid_df, valid_Y, fsets, cfg)

        if cfg.get("early_stopping", True):
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', 
                patience=cfg['patience'], restore_best_weights=True)]
        else:
            callbacks = []
        
        if cfg['epochs'] > 0:
            model.fit_generator(train_iterator(),
                steps_per_epoch=np.ceil(train_df.shape[0] / cfg['batch_size']),
                epochs=cfg['epochs'],
                verbose=cfg['verbose'],
                validation_data=valid_iterator(),
                validation_steps=np.ceil(valid_df.shape[0] / cfg['batch_size']),
                callbacks=callbacks)

        if cfg.get("trained_model_path", None) is not None:
            print("Saving model")
            model.save_weights(cfg["trained_model_path"])

        if cfg.get("save_tjs", False):
            tfjs.converters.save_keras_model(model, cfg["tjs_path"])

    else:
        
        model.load_weights(cfg["trained_model_path"]).expect_partial()
    
    if return_model:
        return model, fsets
    

    test_iterator = create_data_iterator(test_df, test_Y, fsets, cfg, False)
        
    preds = model.predict(test_iterator(), steps=np.ceil(test_df.shape[0] / cfg['batch_size']))
    y_target = np.argmax(test_Y, axis=1)


    ix = np.sum(np.isnan(preds), axis=1) > 0
    
    r, cm = eval_funcs.eval_classifier(y_target, preds)
    
    if print_results:
        eval_funcs.print_eval_classifier(r)
    

    weights_list = output_layer.get_weights()

    np.savez(output_path,
            preds = preds,
            y_target = y_target,
            cfg = cfg, 
            r=r,
            cm=cm,
            rep=rep,
            weights=weights_list[0],
            biases=weights_list[1],
            labels=feature_labels,
            fold=fold)

    return None 

def load_features(cfg):
    feature_labels = []
    processors = []

    for spec in cfg['spec']:
        klass = globals()[spec['processor']]
        proc = klass(spec)
        feature_labels.extend(proc.feature_labels)
        processors.append(proc)

    return processors, feature_labels

def create_data_iterator(df, y, processors, cfg, shuffle=True):
    idx = np.arange(df.shape[0])
    batch_size = cfg['batch_size']


    def iterator():
        while True:
            if shuffle:
                rng.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                indecies = idx[i:(i+batch_size)]
                
                batch_df = df.iloc[indecies]
                batch_y = y[indecies,:]
                
                features = []

                for proc in processors:
                    features.append(proc.transform(batch_df))
                
                batch_F = np.hstack(features)
                
                if np.sum(np.isnan(batch_F)) > 0:
                    print(batch_F)
                    print(batch_F.shape)
                yield (batch_F, batch_y)
    
    return iterator

class BinnedSmfProcessor(object):

    def __init__(self, cfg):
        d_smf = np.load(cfg['path'], allow_pickle=True)
        self.F = d_smf['F']
        self.feature_labels = []

        orig_feature_labels = d_smf['feature_labels']
        for i in range(len(orig_feature_labels)):
            for j in range(i, len(orig_feature_labels)):
                for k in range(j, len(orig_feature_labels)):
                    self.feature_labels.append('smf_%s%s%s' % (orig_feature_labels[i], orig_feature_labels[j], orig_feature_labels[k]))

    def transform(self, df):
        smf_features = []
        F = self.F 

        for u in range(F.shape[1]):
            for w in range(u, F.shape[1]):
                for l in range(w, F.shape[1]):
                    smf_features.append(
                        (F[df['a_id'], u] * F[df['b_id'], w] * F[df['c_id'], l]).astype(bool) | 
                        (F[df['a_id'], u] * F[df['b_id'], l] * F[df['c_id'], w]).astype(bool) | 
                        (F[df['a_id'], w] * F[df['b_id'], u] * F[df['c_id'], l]).astype(bool) |  
                        (F[df['a_id'], w] * F[df['b_id'], l] * F[df['c_id'], u]).astype(bool) | 
                        (F[df['a_id'], l] * F[df['b_id'], u] * F[df['c_id'], w]).astype(bool) |  
                        (F[df['a_id'], l] * F[df['b_id'], w] * F[df['c_id'], u]).astype(bool)
                    )
                
        smf_features = np.vstack(smf_features).T.astype(int)
        return smf_features
    
class GoProcessorSum(object):

    def __init__(self, cfg):
        d_go = np.load(cfg['path'])

        self.F = d_go['F']

        self.feature_labels = []

        self.feature_labels.extend(['sgo_sum_%s' % (s) for s in d_go['feature_labels']])
        
    def transform(self, df):
        F = self.F 

        go_a = F[df['a_id'], :]
        go_b = F[df['b_id'], :]
        go_c = F[df['c_id'], :]

        return go_a + go_b + go_c


class PairwiseProcessor(object):

    def __init__(self, cfg):
        F = np.load(cfg['path'])

        if cfg['normalize']:
            iu = np.triu_indices(F.shape[0], 1)
            vals = F[:,:][iu]
            mu = np.mean(vals)
            std = np.std(vals, ddof=1)
            F = (F - mu) / std 

        self.F = F 
        self.mu = mu 
        self.std = std
        self.feature_labels = ["Shortest Circuit Length"]
    
    def transform(self, df):
        return self.F[df['a_id'], df['b_id'], np.newaxis] + \
                    self.F[df['a_id'], df['c_id'], np.newaxis] + \
                    self.F[df['b_id'], df['c_id'], np.newaxis]


class StandardProcessor(object):

    def __init__(self, cfg):
        d = np.load(cfg['path'])
        ix = d['feature_labels'] == cfg['feature']
        self.F = d['F'][:, ix]

        print(self.F.shape) 
        self.feature_labels = ['sum_%s' % cfg['feature']]

    def transform(self, df):
        sum_F = self.F[df['a_id']] + self.F[df['b_id']] + self.F[df['c_id']]
        return sum_F
        
        
def weighted_categorical_xentropy(y_true, y_pred):
    
    xe = y_true * tf.math.log(y_pred)

    # (Kx1)
    xe = tf.reduce_sum(xe, axis=1, keepdims=True)

    # (1xC)
    class_freq = tf.reduce_sum(y_true, axis=0, keepdims=True)
    
    # (KxC) * (Cx1) = (Kx1)
    weights = tf.matmul(y_true, class_freq, transpose_b=True)

    return -tf.reduce_sum(xe / weights)

if __name__ == "__main__":
    
    cfg_path = sys.argv[1]
    rep = int(sys.argv[2])
    fold = int(sys.argv[3])
    output_path = sys.argv[4]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    main(cfg, rep, fold, output_path)