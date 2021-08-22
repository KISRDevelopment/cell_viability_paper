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
import scipy.sparse
from termcolor import colored

def main(cfg, rep, fold, output_path, print_results=True):

    dataset_path = cfg['task_path']
    train_test_path = cfg['splits_path']

    # load dataset
    df = pd.read_csv(dataset_path)
    
    # create output
    Y = keras.utils.to_categorical(df[cfg['target_col']])
    print(Y.shape)
    # load train/test split 
    data = np.load(train_test_path)
    train_sets = data['train_sets']
    valid_sets = data['valid_sets']
    test_sets = data['test_sets']

    # create training and testing data frames
    train_ix = train_sets[rep, fold,:]
    valid_ix = valid_sets[rep,fold,:]
    test_ix = test_sets[rep,fold,:]

    if cfg.get("train_on_full_dataset", False):
        print(colored("******** TRAINING ON FULL DATASET ***********", "red"))
        train_ix = train_ix + test_ix
    
    if cfg.get("test_on_full_dataset", False):
        print(colored("******** TESTING ON FULL DATASET ***********", "green"))
        test_ix = train_ix + test_ix + valid_ix
        
    train_df = df.iloc[train_ix]
    train_genes = set(train_df['a_id']) | set(train_df['b_id'])

    valid_df = df.iloc[valid_ix]
    valid_genes = set(valid_df['a_id']) | set(valid_df['b_id'])
    print("in Valid but not train: %d" % len(valid_genes - train_genes))

    test_df = df.iloc[test_ix]

    train_X = [np.array(train_df['a_id']), np.array(train_df['b_id'])]
    valid_X = [np.array(valid_df['a_id']), np.array(valid_df['b_id'])]
    test_X = [np.array(test_df['a_id']), np.array(test_df['b_id'])]
    train_Y = Y[train_ix,:]
    valid_Y = Y[valid_ix,:]
    print("Validation sums:")
    print(np.sum(valid_Y, axis=0))
    test_Y = Y[test_ix, :]

    #
    # NN definition
    #

    # single gene features
    n_genes = np.maximum(np.max(df['a_id']), np.max(df['b_id'])) + 1
   
    emb = tf.keras.layers.Embedding(n_genes, cfg['embedding_size'])

    input_a = tf.keras.layers.Input(shape=(1,))
    input_b = tf.keras.layers.Input(shape=(1,))

    embd_a = emb(input_a)
    embd_b = emb(input_b)
    
    merged = embd_a + embd_b
    merged = tf.squeeze(merged, axis=1)
    output_node = layers.Dense(Y.shape[1], activation='softmax')(merged)

    if cfg['balanced_loss']:
        loss = weighted_categorical_xentropy
    else:
        loss = 'categorical_crossentropy'
    
    model = keras.models.Model(inputs=[input_a, input_b], outputs=output_node)
    model.compile(cfg['optimizer'], loss)
    print(model.summary())
    #exit()
    model.outputs[0]._uses_learning_phase = True
    
    # train
    if cfg.get("train_model", True):
        
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg['patience'], restore_best_weights=True)]

        print("Batch size: %d" % int(cfg['batch_size_p'] * train_Y.shape[0]))
        model.fit(
            train_X,
            train_Y,
            epochs=cfg['epochs'],
            verbose=cfg['verbose'],
            validation_data=(valid_X, valid_Y),
            batch_size=int(cfg['batch_size_p'] * train_Y.shape[0]),
            callbacks=callbacks
        )
       
        if cfg.get("trained_model_path", None) is not None:
            print("Saving model")
            model.save_weights(cfg["trained_model_path"])

    else:
        model.load_weights(cfg["trained_model_path"]).expect_partial()
    
    preds = model.predict(test_X)
    y_target = np.argmax(test_Y, axis=1)

    r, cm = eval_funcs.eval_classifier(y_target, preds)
    
    if print_results:
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
    
    cfg_path = sys.argv[1]
    rep = int(sys.argv[2])
    fold = int(sys.argv[3])
    output_path = sys.argv[4]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    main(cfg, rep, fold, output_path)