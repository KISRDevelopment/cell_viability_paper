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
import uuid

import utils.eval_funcs as eval_funcs

def main():
    cfg_path = sys.argv[1]
    rep = int(sys.argv[2])
    fold = int(sys.argv[3])
    output_path = sys.argv[4]

    # load model configuration
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    dataset_path = cfg['task_path']
    targets_path = cfg['targets_path']
    train_test_path = cfg['splits_path']
    
    # load dataset
    df = pd.read_csv(dataset_path)
    
    # create output
    Y = keras.utils.to_categorical(np.load(targets_path)['y'])

    # read features
    all_features, labels = read_features(cfg)
    
    # load train/test split 
    data = np.load(train_test_path)
    train_sets = data['train_sets']
    valid_sets = data['valid_sets']
    test_sets = data['test_sets']

    # create training and testing data frames
    train_ix = train_sets[rep, fold,:]
    valid_ix = valid_sets[rep,fold,:]
    test_ix = test_sets[rep,fold,:]

    if cfg['train_on_full_dataset']:
        print("WARNING: TRAINING ON FULL DATASET --------------------------------------- ")
        # combine training and testing sets
        train_ix = train_ix + test_ix
    
    train_df = df.iloc[train_ix]
    valid_df = df.iloc[valid_ix]
    test_df = df.iloc[test_ix]

    train_Y = Y[train_ix,:]
    valid_Y = Y[valid_ix,:]
    test_Y = Y[test_ix, :]

    
    F = all_features[df['id'], :]
    train_F = F[train_ix, :]
    valid_F = F[valid_ix, :]
    test_F = F[test_ix, :]
    
    # ordinal model
    input_node = layers.Input(shape=(F.shape[1],))
    linear_layer = layers.Dense(1, activation='linear')
    latent_variable = linear_layer(input_node)
    ordinal_layer = OrdinalLayer(train_Y.shape[1])
    output_node = ordinal_layer(latent_variable)

    if cfg['balanced_loss']:
        loss = weighted_categorical_xentropy
    else:
        loss = 'categorical_crossentropy'
    
    model = keras.models.Model(inputs=input_node, outputs=output_node)
    
    model.compile(cfg['optimizer'], loss=loss)

    # setup early stopping
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', 
        patience=cfg['patience'], restore_best_weights=True)]

    # train
    model.fit(
        x=train_F,
        y=train_Y,
        batch_size=cfg['batch_size'],
        epochs=cfg['epochs'],
        verbose=cfg['verbose'],
        validation_data=(valid_F, valid_Y),
        callbacks=callbacks)
    
    preds = model.predict(test_F)
    y_target = np.argmax(test_Y, axis=1)

    r, cm = eval_funcs.eval_classifier(y_target, preds)
    
    eval_funcs.print_eval_classifier(r)

    np.savez(output_path,
        preds = preds,
        y_target = y_target,
        cfg = cfg, 
        r=r,
        cm=cm,
        rep=rep,
        biases=linear_layer.get_weights()[1],
        weights=linear_layer.get_weights()[0],
        thresholds=ordinal_layer.get_thresholds(),
        labels=labels,
        fold=fold)

def read_features(cfg):

    Fs = []
    labels = []
    for feature_spec in cfg['spec']:

        d = np.load(feature_spec['path'])
        F = d['F']
        feature_labels = d['feature_labels']

        if feature_spec['selected_features'] is not None:
            selected_features = set(feature_spec['selected_features'])
            selected_fids = [fid for fid in range(F.shape[1]) if feature_labels[fid] in selected_features]
            F = F[:, selected_fids]
            
            feature_labels = feature_labels[selected_fids]
        
        Fs.append(F)
        labels.extend(feature_labels)
    
    return np.hstack(Fs), labels 

def weighted_categorical_xentropy(y_true, y_pred):
    
    xe = y_true * tf.math.log(y_pred)

    # (Kx1)
    xe = tf.reduce_sum(xe, axis=1, keepdims=True)

    # (1xC)
    class_freq = tf.reduce_sum(y_true, axis=0, keepdims=True)
    # (KxC) * (Cx1) = (Kx1)
    weights = tf.matmul(y_true, class_freq, transpose_b=True)
    
    return -tf.reduce_sum(xe / weights)

class OrdinalLayer(layers.Layer):
    def __init__(self, n_classes, **kwargs):
        self.n_classes = n_classes
        super(OrdinalLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):

        self.deltas = tf.exp(self.add_weight(name='deltas',
                                 shape=(self.n_classes-1,),
                                 initializer=kinit.RandomUniform(minval=-1, maxval=1),
                                 trainable=True))
        self.offset = self.add_weight(name='offset', shape=(1,), initializer=kinit.RandomUniform(minval=-2, maxval=-1), trainable=True)

        self.thresholds = tf.math.cumsum(self.deltas) + self.offset

        super(OrdinalLayer, self).build(input_shape)

    def get_thresholds(self):
        deltas, offset = self.get_weights()
        return np.cumsum(np.exp(deltas)) + offset

    def call(self, x):
        """ X is (Bx1) """
        
        #self.thresholds = tf.Print(self.thresholds, [self.thresholds])
        # (B x n_classes-1)
        diff = self.thresholds - x 
        prob_less_than_i = tf.sigmoid(diff)

        #prob_less_than_i = tf.Print(prob_less_than_i, [prob_less_than_i])
        # (B x n_classes+1)
        prob_less_than_i = tf.concat((tf.zeros_like(x), prob_less_than_i, tf.ones_like(x)), axis=1)
        
        # compute class probabilities (B x n_classes)
        class_probs = prob_less_than_i[:,1:] - prob_less_than_i[:,:-1]
        
        #class_probs = tf.Print(class_probs, [tf.reduce_sum(class_probs, axis=1)])

        return class_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_classes)



if __name__ == "__main__":
    main()