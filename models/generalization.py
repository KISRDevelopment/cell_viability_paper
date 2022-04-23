import json 
import pandas as pd
import numpy as np
import models.nn_single
import models.nn_double
import models.nn_triple
import models.mn 
import models.common

MODELS = {
    "nn_single" : models.nn_single.SingleInputNNModel,
    "nn_double" : models.nn_double.DoubleInputNNModel,
    "nn_triple" : models.nn_triple.TripleInputNNModel,
    "mn" : models.mn.MnModel
}

def train(model_spec, n_workers, train_df, valid_df, output_path, **kwargs):
    model_class = MODELS[model_spec['class']]
    models = []

    for i in range(n_workers):
        model = model_class(model_spec, **kwargs)
        model.train(train_df, valid_df)
        model.save("%s_%d.npz" % (output_path, i))

def evaluate(model_spec, n_workers, output_path, test_df, **kwargs):
    model_class = MODELS[model_spec['class']]
    preds = []
    for i in range(n_workers):
        model = model_class.load("%s_%d.npz" % (output_path, i), **kwargs)
        preds.append(model.predict(test_df, training_norm=False))
    preds = np.array(preds)
    preds = np.mean(preds, axis=0)

    target_col = model_spec['target_col']
    r = models.common.evaluate(np.array(test_df[target_col]), preds)

    return r

def main():

    with open('cfgs/smf_mn_model.json', 'r') as f:
        model_spec = json.load(f)
    
    model_spec['target_col'] = 'is_viable'

    df = pd.read_feather('../generated-data/dataset_yeast_smf.feather')

    splits = np.load("../generated-data/splits/task_yeast_smf_30.npz")['splits']

    train_df, valid_df, _ = models.common.get_dfs(df, splits[0], train_ids=[0, 1, 3], valid_ids=[2], test_ids=[0])

    #train(model_spec, 10, train_df, valid_df, "../tmp/yeast_mn_model")

    test_df = pd.read_feather('../generated-data/dataset_pombe_smf.feather')

    r = evaluate(model_spec, 10, "../tmp/yeast_mn_model", test_df)
    print(r)

if __name__ == "__main__":
    main()