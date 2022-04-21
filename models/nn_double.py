import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np
import json 
import pandas as pd 
import sklearn.metrics 
import models.nn_single

def train_model(model_spec, smf_df, train_df, valid_df):
    sgs = model_spec['single_gene_spec']
    dgs = model_spec['double_gene_spec']

    model = create_model(model_spec)
    
    # prepare single gene inputs
    single_gene_inputs = create_smf_inputs(sgs, smf_df)
    
    # prepare pairwise inputs
    train_double_gene_inputs = create_inputs(dgs, train_df)
    train_double_gene_inputs, mus, stds = normalize_inputs(dgs, train_double_gene_inputs)
    valid_double_gene_inputs = create_inputs(dgs, valid_df)
    valid_double_gene_inputs, _, _ = normalize_inputs(dgs, valid_double_gene_inputs, mus, stds)
    
    # outputs
    train_Y = keras.utils.to_categorical(train_df[model_spec['target_col']])
    valid_Y = keras.utils.to_categorical(valid_df[model_spec['target_col']])

    earlystopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                patience=model_spec['patience'], restore_best_weights=True)
    callbacks = [earlystopping_callback]

    # create data iterators (necessary because some feature sets are too large to put in ram)
    train_iterator = create_data_iterator(train_df, train_Y, single_gene_inputs, train_double_gene_inputs, model_spec['batch_size'])
    valid_iterator = create_data_iterator(valid_df, valid_Y, single_gene_inputs, valid_double_gene_inputs, model_spec['batch_size']*10, False)

    model.fit(x=train_iterator(),
              steps_per_epoch=np.ceil(train_df.shape[0] / model_spec['batch_size']),
              epochs=model_spec['epochs'],
              verbose=model_spec['verbose'],
              validation_data=valid_iterator(),
              validation_steps=np.ceil(valid_df.shape[0] / model_spec['batch_size']),
              callbacks=callbacks)

    return model, mus, stds, single_gene_inputs


def create_smf_inputs(model_spec, df):

    inputs = []
    for feature_set in model_spec['selected_feature_sets']:
        props = model_spec['feature_sets'][feature_set]
        F = np.array(df[ props['cols'] ])
        
        if props['normalize']:
            mu = np.mean(F, axis=0)
            std = np.std(F, axis=0, ddof=1)+1e-9
            F = (F - mu) / std
        
        fdf = pd.DataFrame(data=F, index=df['id'], columns=props['cols'])
        
        inputs.append(fdf)
    
    return inputs


def create_inputs(model_spec, df):

    inputs = []
    for feature_set in model_spec['selected_feature_sets']:
        props = model_spec['feature_sets'][feature_set]
        F = np.array(df[ props['cols'] ])
        inputs.append(F)
    
    return inputs

def normalize_inputs(model_spec, inputs, mus = None, stds = None):

    if mus is None:
        mus = []
        stds = []
        for F in inputs:
            mu = np.mean(F, axis=0)
            std = np.std(F, axis=0, ddof=1)

            min_F = np.min(F, axis=0)
            max_F = np.max(F, axis=0)
            between_zero_and_one = (min_F >= 0) & (max_F <= 1)
            normalize = ~between_zero_and_one
            
            mu = mu * normalize
            std = std * normalize + (1-normalize)

            mus.append(mu)
            stds.append(std)
    
    normalized_inputs = []
    for feature_set, F, mu, std in zip(model_spec['selected_feature_sets'], inputs, mus, stds):
        props = model_spec['feature_sets'][feature_set]
        
        F = (F - mu) / std

        normalized_inputs.append(F)
        
    return normalized_inputs, mus, stds

def create_model(model_spec):
    output_dim = model_spec['n_output_dim']

    sgs = model_spec['single_gene_spec']
    dgs = model_spec['double_gene_spec']

    single_gene_emb_module = models.nn_single.create_single_gene_embedding_module(sgs)
    double_gene_emb_module = models.nn_single.create_single_gene_embedding_module(dgs)

    inputs_a = [keras.layers.Input(shape=(sgs['feature_sets'][fs]['dim'],)) for fs in sgs['selected_feature_sets']]
    inputs_b = [keras.layers.Input(shape=(sgs['feature_sets'][fs]['dim'],)) for fs in sgs['selected_feature_sets']]
    inputs_ab = [keras.layers.Input(shape=(dgs['feature_sets'][fs]['dim'],)) for fs in dgs['selected_feature_sets']]

    output_a = single_gene_emb_module(inputs_a)
    output_b = single_gene_emb_module(inputs_b)
    output_ab = double_gene_emb_module(inputs_ab)

    merged = keras.layers.Concatenate()([(output_a + output_b)/2, output_ab])
    output_layer = keras.layers.Dense(output_dim, activation='softmax')(merged)

    model = keras.Model(inputs=inputs_a + inputs_b + inputs_ab, outputs=output_layer)
    opt = keras.optimizers.Nadam(learning_rate=model_spec['learning_rate'])
    model.compile(opt, loss=models.nn_single.weighted_categorical_xentropy)
    #print(model.summary())

    return model

def create_data_iterator(df, Y, single_fsets, pairwise_fsets, batch_size, training=True):
    idx = np.arange(df.shape[0])
    
    def iterator():
        while True:
            if training:
                np.random.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                indecies = idx[i:(i+batch_size)]
                
                batch_df = df.iloc[indecies]
                batch_Y = Y[indecies,:]

                a_id = batch_df['a_id']
                b_id = batch_df['b_id']

                inputs_a = [np.array(fs.loc[a_id]) for fs in single_fsets]
                inputs_b = [np.array(fs.loc[b_id]) for fs in single_fsets]
                inputs_ab = [fs[indecies,:] for fs in pairwise_fsets] 

                inputs = inputs_a + inputs_b + inputs_ab
                yield (inputs, batch_Y)
    
    return iterator 

def train_and_evaluate_model(model_spec, smf_df, df, split, model_output_path=None, train_ids=[1], valid_ids=[2], test_ids=[3]):
    
    add_extra_info_to_spec(model_spec, smf_df, df)
    
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

    model, mus, stds, single_gene_inputs = train_model(model_spec, smf_df, train_df, valid_df)
    
    if model_output_path is not None:
        weights = model.get_weights()
        np.savez(model_output_path,
            model_spec=model_spec, 
            weights=np.array(weights, dtype=object), 
            mus=np.array(mus, dtype=object), 
            stds=np.array(stds, dtype=object))

    r = evaluate_model(model_spec, model, single_gene_inputs, test_df, mus, stds)

    return r 

# def evaluate_saved_model(saved_model_path, smf_df, df):

#     d = np.load(saved_model_path, allow_pickle=True)
#     weights = d['weights']
#     mus = d['mus'].tolist()
#     stds = d['stds'].tolist()
#     model_spec = d['model_spec'].item()
    
#     model = create_model(model_spec)
#     model.set_weights(weights)

#     # prepare single gene inputs
#     single_gene_inputs = create_smf_inputs(model_spec['single_gene_spec'], smf_df)
    
#     return evaluate_model(model_spec, model, single_gene_inputs, df)
    

def evaluate_model(model_spec, model, single_gene_inputs, df, mus, stds):
    # prepare double inputs
    dgs = model_spec['double_gene_spec']
    test_double_gene_inputs = create_inputs(dgs, df)
    test_double_gene_inputs, _, _ = normalize_inputs(dgs, test_double_gene_inputs, mus, stds)

    batch_size =  model_spec['batch_size']*10
    test_iterator = create_data_iterator(df, np.zeros((df.shape[0], model_spec['n_output_dim'])), single_gene_inputs, test_double_gene_inputs, batch_size, True)

    preds = model.predict(x=test_iterator(), steps=np.ceil(df.shape[0] / batch_size))
    
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

def add_extra_info_to_spec(model_spec, smf_df, df):

    # calculate output dimension size (number of classes)
    model_spec['n_output_dim'] = np.unique(df[model_spec['target_col']]).shape[0]

    add_extra_info_to_subspec(model_spec['single_gene_spec'], smf_df)
    add_extra_info_to_subspec(model_spec['double_gene_spec'], df)
    
def add_extra_info_to_subspec(model_spec, df):
    if len(model_spec['selected_feature_sets']) == 0:
        model_spec['selected_feature_sets'] = list(model_spec['feature_sets'].keys())
        
    # add the feature dimensions and column names for each feature set
    for feature_set, props in model_spec['feature_sets'].items():
        ix = df.columns.str.startswith('%s-' % feature_set)
        props['dim'] = np.sum(ix)
        props['cols'] = list(df.columns[ix])


if __name__ == "__main__":
    import sys 

    model_spec_path = sys.argv[1]
    smf_dataset_path = sys.argv[2]
    dataset_path = sys.argv[3]
    splits_path = sys.argv[4]
    split = int(sys.argv[5])
    model_output_path = sys.argv[6]

    with open(model_spec_path, 'r') as f:
        model_spec = json.load(f)
    
    smf_df = pd.read_feather(smf_dataset_path)
    df = pd.read_feather(dataset_path)

    splits = np.load(splits_path, allow_pickle=True)['splits']
    split = splits[split]

    train_and_evaluate_model(model_spec, smf_df, df, split, model_output_path)

