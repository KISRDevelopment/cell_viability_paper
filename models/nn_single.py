import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np
import json 
import pandas as pd 
def train_model(model_spec, train_df, valid_df):
    if len(model_spec['selected_feature_sets']) == 0:
        model_spec['selected_feature_sets'] = list(model_spec['feature_sets'].keys())
    
    df_cols = train_df.columns 
    
    for feature_set, props in model_spec['feature_sets'].items():
        ix = df_cols.str.startswith('%s-' % feature_set)
        props['dim'] = np.sum(ix)
        props['cols'] = df_cols[ix]


    train_Y = keras.utils.to_categorical(train_df[model_spec['target_col']])
    valid_Y = keras.utils.to_categorical(valid_df[model_spec['target_col']])

    model = create_model(model_spec, train_Y.shape[1])

    train_inputs = create_inputs(model_spec, train_df)
    model.fit(train_inputs, train_Y, epochs=100, verbose=True)

def create_inputs(model_spec, df):

    inputs = []
    for feature_set, props in model_spec['feature_sets'].items():
        F = np.array(df[ props['cols'] ])
        
        if props['normalize']:
            mu = np.mean(F, axis=0)
            std = np.std(F, axis=0, ddof=1)
            F = (F - mu) / std 
        
        inputs.append(F)
    
    return inputs

def create_model(model_spec, output_dim):
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

    # output
    output_layer = keras.layers.Dense(output_dim, activation='softmax')(emb_layer)

    model = keras.Model(inputs=input_layers, outputs=output_layer)
    opt = keras.optimizers.Nadam(learning_rate=model_spec['learning_rate'])
    model.compile(opt, loss=weighted_categorical_xentropy)
    #print(model.summary())

    return model

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

def main(spec_path, dataset_path, splits_path, rep):
    rep = int(rep)

    with open(spec_path, 'r') as f:
        model_spec = json.load(f)

    df = pd.read_csv(dataset_path)

    split = get_split(splits_path, rep)

    train_ix = split == 1
    valid_ix = split == 2
    test_ix = split == 3

    train_df = df[train_ix]
    valid_df = df[valid_ix]
    test_df = df[test_ix]

    train_model(model_spec, train_df, test_df)

def get_split(splits_path, rep):

    d = np.load(splits_path)
    splits = d['splits']
    return splits[rep,:]
    
if __name__ == "__main__":
    import sys 
    main(*sys.argv[1:])
