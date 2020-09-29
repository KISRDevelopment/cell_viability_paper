import numpy as np 
import os
import sys 
import utils.eval_funcs
import uuid 

def main(model_module, cfg, model_files, output_path):

    # we're gonna use a pre-trained model so no need to train
    cfg['train_model'] = False 

    result_files = []
    for i, model_file in enumerate(model_files):
        cfg['trained_model_path'] = model_file
        result_files.append("../tmp/%s.npz" % str(uuid.uuid4()))
        print("Model %d" % i)
        model_module.main(cfg, 0, 0, result_files[-1], print_results=True)
    
    preds = []
    y_target = None 
    for file in result_files:
        d = np.load(file, allow_pickle=True)
        preds.append(d['preds'])
        y_target = d['y_target']

    preds = np.array(preds)

    preds = np.mean(preds, axis=0)

    r, cm = utils.eval_funcs.eval_classifier(y_target, preds)
    
    print("ENSEMBLE RESULT:")
    utils.eval_funcs.print_eval_classifier(r)
    
    np.savez(output_path,
        preds = preds,
        y_target = y_target,
        cfg = cfg, 
        r=r,
        cm=cm,
        rep=0,
        fold=0)
    
