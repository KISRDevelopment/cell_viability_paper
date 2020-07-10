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

import sklearn.metrics
import scipy.stats as stats 
import numpy.random as rng
import uuid

import models.feature_loader as feature_loader
import models.nn_arch as nn_arch
import utils.eval_funcs as eval_funcs

def main():
    cfg_path = sys.argv[1]
    dataset_path = sys.argv[2]
    targets_path = sys.argv[3]
    train_test_path = sys.argv[4]
    rep = int(sys.argv[5])
    fold = int(sys.argv[6])
    output_path = sys.argv[7]

    # load model configuration
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    # load dataset
    df = pd.read_csv(dataset_path)
    

    # create output
    Y = keras.utils.to_categorical(np.load(targets_path)['y'])
    
    # load input features
    single_gene_spec = [s for s in cfg['spec'] if not s['pairwise']]
    single_fsets = feature_loader.load_feature_sets(single_gene_spec, scramble=False)
    
    # load train/test split 
    data = np.load(train_test_path)
    train_sets = data['train_sets']
    valid_sets = data['valid_sets']
    test_sets = data['test_sets']

    # create training and testing data frames
    train_ix = train_sets[rep, fold,:]
    valid_ix = valid_sets[rep,fold,:]
    test_ix = test_sets[rep,fold,:]

    train_df = df.iloc[train_ix]
    valid_df = df.iloc[valid_ix]
    test_df = df.iloc[test_ix]

    train_Y = Y[train_ix,:]
    valid_Y = Y[valid_ix,:]
    test_Y = Y[test_ix, :]

    # setup feature sets
    train_fsets = [single_fsets[i][train_df['id'], :] for i in range(len(single_fsets))]
    if cfg['scramble']:
        train_fsets = [f[rng.permutation(f.shape[0]),:] for f in train_fsets]
    valid_fsets = [single_fsets[i][valid_df['id'], :] for i in range(len(single_fsets))]
    test_fsets = [single_fsets[i][test_df['id'], :] for i in range(len(single_fsets))]

    single_fsets_shapes = [F.shape[1:] for F in train_fsets]
    

    #
    # NN definition
    #

    # single gene features
    inputs_a = nn_arch.create_input_nodes(single_gene_spec, single_fsets_shapes, base_name="a")
    single_gene_arch = nn_arch.create_input_architecture(output_size=cfg['embedding_size'], 
        output_activation=cfg['embedding_activation'], 
        name='single_input', 
        spec=single_gene_spec)
    output_a = single_gene_arch(inputs_a, name="input_a")
    output_node = layers.Dense(Y.shape[1], activation='softmax')(output_a)

    if cfg['balanced_loss']:
        loss = weighted_categorical_xentropy
    else:
        loss = 'categorical_crossentropy'
    
    model = keras.models.Model(inputs=inputs_a, outputs=output_node)
    model.compile(cfg['optimizer'], loss=loss)

    # setup early stopping
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', 
        patience=cfg['patience'], restore_best_weights=True)]

    # train
    model.fit(
        x=train_fsets,
        y=train_Y,
        batch_size=cfg['batch_size'],
        epochs=cfg['epochs'],
        verbose=cfg['verbose'],
        validation_data=(valid_fsets, valid_Y),
        callbacks=callbacks)
    
    preds = model.predict(test_fsets)
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
        fold=fold)

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
    main()