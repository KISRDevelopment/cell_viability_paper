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

    inputs = []
    for feature_set in model_spec['selected_feature_sets']:
        props = model_spec['feature_sets'][feature_set]
        F = np.array(df[ props['cols'] ])
        inputs.append(F)
    
    return inputs

def normalize_inputs(model_spec, inputs, mus = None, stds = None):

    if mus is None:
        mus = [np.mean(F, axis=0) for F in inputs]
        stds = [np.std(F, axis=0, ddof=1)+1e-9 for F in inputs]
    
    normalized_inputs = []
    for feature_set, F, mu, std in zip(model_spec['selected_feature_sets'], inputs, mus, stds):
        props = model_spec['feature_sets'][feature_set]

        if props['normalize']:
            F = (F - mu) / std
        normalized_inputs.append(F)
        
    return normalized_inputs, mus, stds

def create_model(model_spec):
    output_dim = model_spec['n_output_dim']

    input_layers = [keras.layers.Input(shape=(model_spec['feature_sets'][fs]['dim'],)) for 
        fs in model_spec['selected_feature_sets']]
    
    emb_layer = create_single_gene_embedding_module(model_spec)(input_layers)
    output_layer = keras.layers.Dense(output_dim, activation='softmax')(emb_layer)

    model = keras.Model(inputs=input_layers, outputs=output_layer)
    opt = keras.optimizers.Nadam(learning_rate=model_spec['learning_rate'])
    model.compile(opt, loss=weighted_categorical_xentropy)
    #print(model.summary())

    return model

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

    train_ix = np.isin(split, train_ids)
    valid_ix = np.isin(split, valid_ids)
    test_ix = np.isin(split, test_ids)

    train_df = df[train_ix]
    valid_df = df[valid_ix]
    test_df = df[test_ix]

    model, mus, stds = train_model(model_spec, train_df, valid_df)
    
    if model_output_path is not None:
        weights = model.get_weights()
        np.savez(model_output_path,
            model_spec=model_spec, 
            weights=np.array(weights, dtype=object), 
            mus=np.array(mus, dtype=object), 
            stds=np.array(stds, dtype=object))


    return evaluate_model(model_output_path, test_df)
    
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

    if len(model_spec['selected_feature_sets']) == 0:
        model_spec['selected_feature_sets'] = list(model_spec['feature_sets'].keys())
    
    # add the feature dimensions and column names for each feature set
    for feature_set, props in model_spec['feature_sets'].items():
        ix = df.columns.str.startswith('%s-' % feature_set)
        props['dim'] = np.sum(ix)
        props['cols'] = list(df.columns[ix])

    # calculate output dimension size (number of classes)
    model_spec['n_output_dim'] = np.unique(df[model_spec['target_col']]).shape[0]

if __name__ == "__main__":
    import sys 

    model_spec_path = sys.argv[1]
    dataset_path = sys.argv[2]
    splits_path = sys.argv[3]
    split = int(sys.argv[4])
    model_output_path = sys.argv[5]

    with open(model_spec_path, 'r') as f:
        model_spec = json.load(f)
    
    df = pd.read_csv(dataset_path)

    splits = np.load(splits_path)['splits']
    split = splits[split,:]

    train_and_evaluate_model(model_spec, df, split, model_output_path)

