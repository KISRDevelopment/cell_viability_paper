import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import sys 
import os
import json 
import utils.eval_funcs as eval_funcs
import pprint 

def main(cfg, rep, fold, output_path):

    X = np.load(cfg['features_path'])
    y = np.load(cfg['targets_path'])['y']
    
    # load train/test split 
    data = np.load(cfg['splits_path'])
    train_sets = data['train_sets']
    valid_sets = data['valid_sets']
    test_sets = data['test_sets']

    # create training and testing data frames
    train_ix = train_sets[rep, fold,:]
    valid_ix = valid_sets[rep,fold,:]
    test_ix = test_sets[rep,fold,:]

    train_ix = train_ix + valid_ix

    Xtrain = X[train_ix, :]
    ytrain = y[train_ix]

    clf = RandomForestClassifier(n_estimators = cfg['n_estimators'], n_jobs=cfg['n_jobs'], verbose=cfg['verbose'], class_weight='balanced')
    clf = clf.fit(Xtrain, ytrain)

    Xtest = X[test_ix, :]
    ytest = y[test_ix]

    yhat = clf.predict(Xtest)
    preds = np.zeros((yhat.shape[0], 2))
    preds[:,0] = yhat 
    preds[:,1] = 1-yhat 

    print(ytest.shape)
    print(preds.shape)

    r, cm = eval_funcs.eval_classifier(ytest, preds)
    pprint.pprint(r)

if __name__ == "__main__":

    cfg_path = sys.argv[1]
    rep = int(sys.argv[2])
    fold = int(sys.argv[3])
    output_path = sys.argv[4]

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    main(cfg, rep, fold, output_path)