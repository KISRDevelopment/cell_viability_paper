import numpy as np 
import pandas as pd 
import tensorflow as tf 
import tensorflow.keras as keras 
import scipy.stats as stats 
import numpy.random as rng 
import utils.eval_funcs 

SLANT_TRAIN_SET = "/home/mmkhajah/Downloads/slant_data_dir/data_dir/training/yeast_training_sl_all.csv"
SLANT_TEST_SET = "/home/mmkhajah/Downloads/slant_data_dir/data_dir/training/yeast_testing_sl_all.csv"
OUTPUT_PATH = "../results/slant_generic_nn.csv"

FEATURES = [
    'betweenness.dist', 'constraint.dist', 'closeness.dist', 'coreness.dist', 'degree.dist', 'eccentricity.dist', 'eigen_centrality.dist', 'hub_score.dist', 'neighborhood1.size.dist', 'neighborhood2.size.dist', 'neighborhood5.size.dist', 'neighborhood6.size.dist',
    'betweenness1', 'constraint1', 'closeness1', 'coreness1', 'degree1', 'eccentricity1', 'eigen_centrality1', 'hub_score1',  'neighborhood2.size1', 'page_rank1', 'betweenness2', 'constraint2', 'closeness2', 
    'coreness2', 'degree2', 'eccentricity2', 'eigen_centrality2', 'hub_score2', 'neighborhood2.size2', 'page_rank2',
    'cohesion', 'mutual_neighbours', 'shortest_path',
    'shared_go_count_p', 'shared_go_count_f', 'shared_go_count_c'
]
OUTPUT_COL = 'sl'
cfg = {
    'n_hidden' : 10,
    'verbose' : True,
    'patience' : 50,
    'n_epochs' : 1000,
    'batch_prop' : 0.1
}

train_df = pd.read_csv(SLANT_TRAIN_SET, sep='\t')

Xtrain = np.array(train_df[FEATURES])
Ytrain = np.array(train_df[OUTPUT_COL] == 'X1').astype(int)

ix = rng.permutation(Xtrain.shape[0])
Xtrain = Xtrain[ix,:]
Ytrain = Ytrain[ix]

# normalize training
mu = np.mean(Xtrain, axis=0, keepdims=True)
std = np.std(Xtrain, axis=0, ddof=1, keepdims=True)
Xtrain = (Xtrain - mu) / std 

input_layer = keras.layers.Input(shape=(Xtrain.shape[1],))
hidden_layer = keras.layers.Dense(cfg['n_hidden'], activation='tanh')(input_layer)
output_layer = keras.layers.Dense(1, activation='sigmoid')(hidden_layer)

model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='nadam', loss='binary_crossentropy')
        
# configure early stopping
callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
    patience=cfg['patience'],
    restore_best_weights=True)

# train
history = model.fit(Xtrain, 
        Ytrain, 
        verbose=cfg['verbose'],
        callbacks=[callback], 
        validation_split=0.2,
        epochs=cfg['n_epochs'],
        batch_size=int(cfg['batch_prop'] * Xtrain.shape[0]))


test_df = pd.read_csv(SLANT_TEST_SET, sep='\t')

Xtest = np.array(test_df[FEATURES])
Ytest = np.array(test_df[OUTPUT_COL] == 'X1').astype(int)

# normalize test
Xtest = (Xtest - mu) / std 

ypred = model.predict(Xtest)[:,0]
yhat = np.zeros((ypred.shape[0], 2))
yhat[:,0] = 1-ypred 
yhat[:,1] = ypred 

eval_results, cm = utils.eval_funcs.eval_classifier(Ytest, yhat)

import pprint
pprint.pprint(eval_results)

output_df = pd.DataFrame(data=yhat, columns=['X0', 'X1'])
output_df.to_csv(OUTPUT_PATH, sep=' ')
