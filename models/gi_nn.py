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

def main(cfg, rep, fold, output_path, print_results=True):

    dataset_path = cfg['task_path']
    targets_path = cfg['targets_path']
    train_test_path = cfg['splits_path']

    # load dataset
    df = pd.read_csv(dataset_path)
    
    # shuffle the order of the pairs to eliminate pathological cases
    A = np.array(df[['a_id', 'b_id']]).T
    np.random.shuffle(A)
    df[['a_id', 'b_id']] = A.T 

    # load input features
    single_gene_spec = [s for s in cfg['spec'] if not s['pairwise']]
    pairwise_gene_spec = [s for s in cfg['spec'] if s['pairwise']]
    single_fsets, single_fsets_shapes = feature_loader.load_feature_sets(single_gene_spec, False)
    pairwise_fsets, pairwise_fsets_shapes = feature_loader.load_feature_sets(pairwise_gene_spec, False)
    #single_fsets_shapes = [F.shape[1:] for F in single_fsets]
    #pairwise_fsets_shapes = [[F.shape[2],] for F in pairwise_fsets]
    
    # create output
    Y = keras.utils.to_categorical(np.load(targets_path)['y'])
    
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

    #
    # NN definition
    #

    # single gene features
    inputs_a = nn_arch.create_input_nodes(single_gene_spec, single_fsets_shapes, base_name="a")
    inputs_b = nn_arch.create_input_nodes(single_gene_spec, single_fsets_shapes, base_name="b")
    single_gene_arch = nn_arch.create_input_architecture(output_size=cfg['embedding_size'], 
        output_activation=cfg['embedding_activation'], 
        name='single_input', spec=single_gene_spec)
    output_a = single_gene_arch(inputs_a, name="input_a")
    output_b = single_gene_arch(inputs_b, name="input_b")

    # pairwise features
    inputs_ab = nn_arch.create_input_nodes(pairwise_gene_spec, pairwise_fsets_shapes, base_name="ab")
    pairwise_gene_arch = nn_arch.create_input_architecture(output_size=cfg['embedding_size'], 
        output_activation=cfg['embedding_activation'], 
        name='pairwise_input', spec=pairwise_gene_spec)
    output_ab = pairwise_gene_arch(inputs_ab, name="input_ab")

    merged = nn_arch.concatenate([output_a, output_b, output_ab], name="preoutput")
    output_node = layers.Dense(Y.shape[1], activation='softmax')(merged)

    if cfg['balanced_loss']:
        loss = weighted_categorical_xentropy
    else:
        loss = 'categorical_crossentropy'
    
    model = keras.models.Model(inputs=inputs_a + inputs_b + inputs_ab, outputs=output_node)
    model.compile(cfg['optimizer'], loss)
    model.outputs[0]._uses_learning_phase = True
    
    # train
    if cfg.get("train_model", True):
        
        # create data iterators (necessary because some feature sets are too large to put in ram)
        train_iterator = create_data_iterator(train_df, train_Y, single_fsets, pairwise_fsets, cfg, cfg['scramble'])
        valid_iterator = create_data_iterator(valid_df, valid_Y, single_fsets, pairwise_fsets, cfg, False)

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg['patience'], restore_best_weights=True)]

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

    else:
        model.load_weights(cfg["trained_model_path"]).expect_partial()
    
    test_F = feature_transform(test_df, single_fsets, pairwise_fsets)
    preds = model.predict(test_F)
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
    
def create_data_iterator(df, y, single_fsets, pairwise_fsets, cfg, scramble=False):
    idx = np.arange(df.shape[0])
    batch_size = cfg['batch_size']

    def iterator():
        while True:
            rng.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                indecies = idx[i:(i+batch_size)]
                
                batch_df = df.iloc[indecies]
                batch_y = y[indecies,:]
                batch_F = feature_transform(batch_df, single_fsets, pairwise_fsets)
                
                if scramble:
                    batch_F = [f[rng.permutation(f.shape[0]),:] for f in batch_F]
                
                yield (batch_F, batch_y)
    
    return iterator 

def feature_transform(df, single_fsets, pairwise_fsets):
    inputs_A = []
    inputs_B = []
    inputs_AB = []
    for fset in single_fsets:
        inputs_A.append(fset[df['a_id'], :])
        inputs_B.append(fset[df['b_id'], :])

    a_id = np.array(df['a_id'])
    b_id = np.array(df['b_id'])


    for fset in pairwise_fsets:
        if type(fset) != dict:
            inputs_AB.append(fset[df['a_id'], df['b_id'], :])
        else:
            first_val = next(iter(fset.values()))
            fset_shape = len(first_val)

            PF = []
            for i in range(df.shape[0]):
                key = tuple(sorted((a_id[i], b_id[i])))
                if key in fset:
                    PF.append(fset[key])
                else:
                    PF.append(np.zeros(fset_shape))
            PF = np.array(PF)
            inputs_AB.append(PF)
            
    return inputs_A + inputs_B + inputs_AB

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