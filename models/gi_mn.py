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

    
    
    fsets, feature_labels = load_features(cfg)

    #
    # NN definition
    #
    input_node = layers.Input(shape=(len(feature_labels),), name='input_features')
    if cfg['type'] == 'orm':
        linear_layer = layers.Dense(1, activation='linear')
        latent_variable = linear_layer(input_node)
        ordinal_layer = OrdinalLayer(Y.shape[1])
        output_node = ordinal_layer(latent_variable)
    else:
        output_layer = layers.Dense(Y.shape[1], activation='softmax')
        output_node = output_layer(input_node)

    if cfg['balanced_loss']:
        loss = weighted_categorical_xentropy
    else:
        loss = 'categorical_crossentropy'
    
    model = keras.models.Model(inputs=input_node, outputs=output_node)
    model.compile(cfg['optimizer'], loss)

    # train
    if cfg.get("train_model", True):
        train_iterator = create_data_iterator(train_df, train_Y, fsets, cfg)
        valid_iterator = create_data_iterator(valid_df, valid_Y, fsets, cfg)

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', 
            patience=cfg['patience'], restore_best_weights=True)]

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
    
    test_iterator = create_data_iterator(test_df, test_Y, fsets, cfg, False)
        
    preds = model.predict(test_iterator(), steps=np.ceil(test_df.shape[0] / cfg['batch_size']))
    y_target = np.argmax(test_Y, axis=1)

    r, cm = eval_funcs.eval_classifier(y_target, preds)
    
    if print_results:
        eval_funcs.print_eval_classifier(r)
    

    if cfg['type'] == 'orm':
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
            labels=feature_labels,
            fold=fold)
    else:

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

def load_features(cfg):
    feature_labels = []

    d_smf = np.load(cfg['smf_path'], allow_pickle=True)
    
    d_go = np.load(cfg['go_path'])
    
    F_spl = np.load(cfg['shortest_path_len_path'])
    iu = np.triu_indices(F_spl.shape[0], 1)
    vals = F_spl[:,:][iu]
    mu = np.mean(vals)
    std = np.std(vals, ddof=1)
    F_spl = (F_spl - mu) / std 

    # d = np.load(cfg['pairwise_comm_path'])
    # indecies = d['indecies']
    # data = d['data']
    # indexed_data = {}
    # for i in range(data.shape[0]):
    #     index = tuple(indecies[i, :])
    #     indexed_data[index] = data[i,:]
    # pairwise_comm = indexed_data

    # feature_labels.extend(['delta_%s' % lbl for lbl in d_smf['feature_labels'][:-1]])
    # feature_labels.extend(['sum_%s' % lbl for lbl in d_smf['feature_labels'][:-1]])
    # feature_labels.append('ess')

    feature_labels.extend(['t' for i in range(6)])

    feature_labels.extend(['sgo_either_%s' % s for s in d_go['feature_labels']])
    feature_labels.extend(['sgo_both_%s' % s for s in d_go['feature_labels']])
    feature_labels.append('shortest_path_len')

    return (d_smf, d_go, F_spl), feature_labels

def create_data_iterator(df, y, fsets, cfg, shuffle=True):
    idx = np.arange(df.shape[0])
    batch_size = cfg['batch_size']

    d_smf, d_go, F_spl = fsets 
    # first_val = next(iter(pcomm.values()))
    # pcomm_shape = len(first_val)

    F_smf = d_smf['F']
    
    F_go = d_go['F']

    def iterator():
        while True:
            if shuffle:
                rng.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                indecies = idx[i:(i+batch_size)]
                
                batch_df = df.iloc[indecies]
                batch_y = y[indecies,:]
                
                features = []
                
                # compute features
                # sum_essentials =  F_smf[batch_df['a_id'], -1] + F_smf[batch_df['b_id'], -1]
                # smf_features = []
                # for i in range(F_smf.shape[1]-1):
                #     delta_smf = np.abs(F_smf[batch_df['a_id'], i] - F_smf[batch_df['b_id'], i])
                #     smf_features.append(delta_smf)

                # for i in range(F_smf.shape[1]-1):
                #     sum_smf = np.abs(F_smf[batch_df['a_id'], i] + F_smf[batch_df['b_id'], i])
                #     smf_features.append(sum_smf)

                # smf_features.append(sum_essentials)
                # smf_features = np.vstack(smf_features).T 

                smf_features = []
                for u in range(F_smf.shape[1]):
                    for w in range(u, F_smf.shape[1]):
                        smf_features.append(
                            (F_smf[batch_df['a_id'], u] * F_smf[batch_df['b_id'], w]).astype(bool) | \
                            (F_smf[batch_df['a_id'], w] * F_smf[batch_df['b_id'], u]).astype(bool))
                
                smf_features = np.vstack(smf_features).T.astype(int)
                # print(np.sum(smf_features, axis=0))

                go_a = F_go[batch_df['a_id'], :]
                go_b = F_go[batch_df['b_id'], :]
                go_either = ((go_a + go_b) > 0).astype(int)
                go_both = go_a * go_b 
                
                spl = F_spl[batch_df['a_id'], batch_df['b_id'],np.newaxis]

                batch_F = np.hstack([smf_features, go_either, go_both, spl])
                
                yield (batch_F, batch_y)
    
    return iterator

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

        self.deltas = self.add_weight(name='deltas',
                                shape=(self.n_classes-1,),
                                initializer=kinit.Constant(np.linspace(-1, 1, self.n_classes-1)),
                                trainable=True)
        self.offset = self.add_weight(name='offset', shape=(1,), 
            initializer=kinit.Constant(0), trainable=True)

        
        super(OrdinalLayer, self).build(input_shape)

    def get_thresholds(self):
        deltas, offset = self.get_weights()
        return np.cumsum(np.exp(deltas)) + offset
        
        #return self.thresholds.numpy()
        
    def call(self, x):
        """ X is (Bx1) """
        
        thresholds = tf.math.cumsum(tf.exp(self.deltas)) + self.offset
        

        # (B x n_classes-1)
        diff = thresholds - x 
        #tf.print("diff: ", diff)
        prob_less_than_i = tf.sigmoid(diff)
        
        #prob_less_than_i = tf.Print(prob_less_than_i, [prob_less_than_i])
        # (B x n_classes+1)
        prob_less_than_i = tf.concat((tf.zeros_like(x), prob_less_than_i, tf.ones_like(x)), axis=1)
        #tf.print("prob_less_than_i: ", prob_less_than_i)

        # compute class probabilities (B x n_classes)
        class_probs = prob_less_than_i[:,1:] - prob_less_than_i[:,:-1]
        #tf.print("class probs: ", class_probs)
        #class_probs = tf.Print(class_probs, [tf.reduce_sum(class_probs, axis=1)])

        return class_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_classes)
        
if __name__ == "__main__":
    
    cfg_path = sys.argv[1]
    rep = int(sys.argv[2])
    fold = int(sys.argv[3])
    output_path = sys.argv[4]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    main(cfg, rep, fold, output_path)