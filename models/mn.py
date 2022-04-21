import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np
import json 
import pandas as pd 
import sklearn.metrics 

def train_model(model_spec, train_df, valid_df):
    
    model = create_model(model_spec)

    train_inputs = create_inputs(model_spec, train_df)
    train_inputs, mus, stds = normalize_inputs(model_spec, train_inputs)
    train_Y = keras.utils.to_categorical(train_df[model_spec['target_col']])
    
    valid_inputs = create_inputs(model_spec, valid_df)
    valid_inputs, _, _ = normalize_inputs(model_spec, valid_inputs, mus, stds)
    valid_Y = keras.utils.to_categorical(valid_df[model_spec['target_col']])

    earlystopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                patience=model_spec['patience'], restore_best_weights=True)
    callbacks = [earlystopping_callback]

    model.fit(train_inputs, 
              train_Y, 
              batch_size=int(model_spec['batch_size_p'] * train_Y.shape[0]),
              epochs=model_spec['epochs'],
              verbose=model_spec['verbose'],
              validation_data=(valid_inputs, valid_Y),
              validation_batch_size=100000,
              callbacks=callbacks)

    return model, mus, stds


def create_inputs(model_spec, df):

    F = np.array(df[model_spec['features']])
    
    return F

def normalize_inputs(model_spec, F, mu = None, std = None):

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

def create_model(model_spec):
    
    input_layer = keras.layers.Input(shape=(len(model_spec['features'],)))
    output_layer = keras.layers.Dense(model_spec['n_output_dim'], activation='softmax')(input_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    opt = keras.optimizers.Nadam(learning_rate=model_spec['learning_rate'])
    model.compile(opt, loss=weighted_categorical_xentropy)
    #print(model.summary())

    return model

def weighted_categorical_xentropy(y_true, y_pred):
    
    xe = y_true * tf.math.log(y_pred)

    # (Kx1)
    xe = tf.reduce_sum(xe, axis=1, keepdims=True)

    # (1xC)
    class_freq = tf.reduce_sum(y_true, axis=0, keepdims=True)
    
    # (KxC) * (Cx1) = (Kx1)
    weights = tf.matmul(y_true, class_freq, transpose_b=True)

    return -tf.reduce_sum(xe / weights)

def train_and_evaluate_model(model_spec, df, split, model_output_path=None, train_ids=[1], valid_ids=[2], test_ids=[3]):
    
    add_extra_info_to_spec(model_spec, df)
    
    train_df, valid_df, test_df = get_dfs(df, split, train_ids, valid_ids, test_ids)

    model, mus, stds = train_model(model_spec, train_df, valid_df)
    
    if model_output_path is not None:
        weights = model.get_weights()
        np.savez(model_output_path,
            model_spec=model_spec, 
            weights=np.array(weights, dtype=object), 
            mus=mus, 
            stds=stds)


    return evaluate_model(model_output_path, test_df)
    
def get_dfs(df, split, train_ids, valid_ids, test_ids):

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

def evaluate_model(saved_model_path, df):

    d = np.load(saved_model_path, allow_pickle=True)
    weights = d['weights']
    mus = d['mus']
    stds = d['stds']
    model_spec = d['model_spec'].item()
    
    model = create_model(model_spec)
    model.set_weights(weights)

    test_inputs = create_inputs(model_spec, df)
    test_inputs, _, _ = normalize_inputs(model_spec, test_inputs, mus, stds)
    
    preds = model.predict(test_inputs, batch_size=1000000)
    
    r = evaluate(np.array(df[model_spec['target_col']]), preds)
    
    return r

def evaluate(ytrue, preds):
    
    yhat = np.argmax(preds, axis=1)

    bacc = sklearn.metrics.balanced_accuracy_score(ytrue, yhat)
    acc = sklearn.metrics.accuracy_score(ytrue, yhat)
    cm = sklearn.metrics.confusion_matrix(ytrue, yhat)

    Ytrue = keras.utils.to_categorical(ytrue)
    auc_roc = sklearn.metrics.roc_auc_score(Ytrue, preds, average=None)

    print("Accuracy: %0.2f" % acc)
    print("Balanced Accuracy: %0.2f" % bacc)
    print("AUC-ROC: ", auc_roc)
    print("Confusion Matrix: ")
    print(cm)

    return {
        "bacc" : bacc,
        "acc" : acc,
        "auc_roc" : auc_roc.tolist(),
        "cm" : cm.tolist()
    }

def load_split(splits_path, rep):

    d = np.load(splits_path)
    splits = d['splits']
    return splits[rep,:]

def add_extra_info_to_spec(model_spec, df):

    # calculate output dimension size (number of classes)
    model_spec['n_output_dim'] = np.unique(df[model_spec['target_col']]).shape[0]

    # calculate actual features
    ix = np.zeros_like(df.columns, dtype=bool)
    for f in model_spec['features']:
        ix = ix | df.columns.str.startswith(f)
    model_spec['features'] = list(df.columns[ix])

if __name__ == "__main__":
    import sys 

    model_spec_path = sys.argv[1]
    dataset_path = sys.argv[2]
    splits_path = sys.argv[3]
    split = int(sys.argv[4])
    model_output_path = sys.argv[5]

    with open(model_spec_path, 'r') as f:
        model_spec = json.load(f)
    
    df = pd.read_feather(dataset_path)

    splits = np.load(splits_path, allow_pickle=True)['splits']
    split = splits[split]
    
    train_and_evaluate_model(model_spec, df, split, model_output_path)

