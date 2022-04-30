import numpy as np 
import pandas as pd 
import sklearn.metrics 
import tensorflow as tf 
import tensorflow.keras as keras 
from scipy import interp

def create_inputs(model_spec, df):

    inputs = []
    for feature_set in model_spec['selected_feature_sets']:
        props = model_spec['feature_sets'][feature_set]
        eff_cols = ['%s' % c for c in props['cols']]
        F = df[ eff_cols ].to_numpy()
        inputs.append(F)
    
    return inputs

def normalize_inputs(inputs, mus = None, stds = None):

    if mus is None:
        mus = [None for inp in inputs]
        stds = [None for inp in inputs]
        
    normalized_inputs = []
    for i, F in enumerate(inputs):
        F, mu, std = normalize(F, mus[i], stds[i])
        mus[i] = mu
        stds[i] = std
        normalized_inputs.append(F)
    
    return normalized_inputs, mus, stds

def normalize(F, mu = None, std = None):

    if mu is None:
        mu = np.mean(F, axis=0)
        std = np.std(F, axis=0, ddof=1)+1e-9

        min_F = np.min(F, axis=0)
        max_F = np.max(F, axis=0)
        between_zero_and_one = (min_F >= 0) & (max_F <= 1)
        normalize = ~between_zero_and_one
        
        mu = mu * normalize
        std = std * normalize + (1-normalize)

    F = (F - mu) / std    
        
    return F, mu, std

def get_dfs(df, split, train_ids=[1], valid_ids=[2], test_ids=[3]):

    if type(split) == dict:
        partition = [split['test_genes'], split['train_genes'], split['valid_genes'], split['dev_test_genes']]

        train_genes = set.union(*[set(partition[i]) for i in train_ids])
        valid_genes = set.union(*[set(partition[i]) for i in valid_ids])
        test_genes = set.union(*[set(partition[i]) for i in test_ids])
        
        train_ix = df['a_id'].isin(train_genes) & df['b_id'].isin(train_genes)
        valid_ix = df['a_id'].isin(valid_genes) & df['b_id'].isin(valid_genes)
        test_ix = df['a_id'].isin(test_genes) & df['b_id'].isin(test_genes)

        train_df = df[train_ix]
        valid_df = df[valid_ix]
        test_df = df[test_ix]
    else:
        train_ix = np.isin(split, train_ids)
        valid_ix = np.isin(split, valid_ids)
        test_ix = np.isin(split, test_ids)
        
        train_df = df[train_ix]
        valid_df = df[valid_ix]
        test_df = df[test_ix]
    
    return train_df, valid_df, test_df 


BASE_FPR = np.linspace(0, 1, 101)

def evaluate(ytrue, preds):
    
    yhat = np.argmax(preds, axis=1)

    bacc = sklearn.metrics.balanced_accuracy_score(ytrue, yhat)
    acc = sklearn.metrics.accuracy_score(ytrue, yhat)
    cm = sklearn.metrics.confusion_matrix(ytrue, yhat)

    Ytrue = keras.utils.to_categorical(ytrue)
    
    # per class
    auc_roc = sklearn.metrics.roc_auc_score(Ytrue, preds, average=None)
    pr = sklearn.metrics.average_precision_score(Ytrue, preds, average=None)
    per_class_bacc = []
    per_class_tpr = []

    for b in range(Ytrue.shape[1]):
        y_bin = ytrue == b
        target_pred = yhat == b
        per_class_bacc.append(sklearn.metrics.balanced_accuracy_score(y_bin, target_pred))

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_bin, preds[:,b])
        tpr = interp(BASE_FPR, fpr, tpr)
        tpr[0] = 0.0

        per_class_tpr.append(tpr.tolist())
    
    # print("Accuracy: %0.2f" % acc)
    # print("Balanced Accuracy: %0.2f" % bacc)
    # print("AUC-ROC: ", auc_roc)
    # print("Confusion Matrix: ")
    # print(cm)

    return {
        "bacc" : bacc,
        "acc" : acc,
        "auc_roc" : auc_roc.tolist(),
        "cm" : cm.tolist(),
        "pr" : pr.tolist(),
        "per_class_bacc" : per_class_bacc,
        "per_class_tpr" : per_class_tpr,
    }


def weighted_categorical_xentropy(y_true, y_pred):
    
    xe = y_true * tf.math.log(y_pred)

    # (Kx1)
    xe = tf.reduce_sum(xe, axis=1, keepdims=True)

    # (1xC)
    class_freq = tf.reduce_sum(y_true, axis=0, keepdims=True)
    
    # (KxC) * (Cx1) = (Kx1)
    weights = tf.matmul(y_true, class_freq, transpose_b=True)

    return -tf.reduce_sum(xe / weights)

def calculate_output_dim(df, col):
    return np.unique(df[col]).shape[0]
