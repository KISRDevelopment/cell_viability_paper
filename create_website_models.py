import numpy as np 
import models.train_and_evaluate
import json 
import copy 
import os 
import pandas as pd 
import models.mn 
import models.null
import concurrent.futures
import models.common 
import glob 

def load_spec(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():

    mn_spec = load_spec("cfgs/gi_mn_model.json")
    mn_spec['target_col'] = 'is_neutral'
    
    """ train GI models """
    train_gi_model(mn_spec, 
                "../generated-data/dataset_yeast_gi_hybrid_mn.feather",
                "../generated-data/splits/dataset_yeast_gi_hybrid.npz",
                "../results/website_models/gi_yeast")
    train_gi_model(mn_spec, 
                "../generated-data/dataset_pombe_gi_mn.feather",
                "../generated-data/splits/dataset_pombe_gi.npz",
                "../results/website_models/gi_pombe")
    train_gi_model(mn_spec, 
                "../generated-data/dataset_human_gi_mn.feather",
                "../generated-data/splits/dataset_human_gi.npz",
                "../results/website_models/gi_human")
    train_gi_model(mn_spec, 
                "../generated-data/dataset_dro_gi_mn.feather",
                "../generated-data/splits/dataset_dro_gi.npz",
                "../results/website_models/gi_dro")
    
    save_model_for_website("../results/website_models/gi_yeast", "website/models/gi_yeast.npz")
    save_model_for_website("../results/website_models/gi_pombe", "website/models/gi_pombe.npz")
    save_model_for_website("../results/website_models/gi_human", "website/models/gi_human.npz")
    save_model_for_website("../results/website_models/gi_dro", "website/models/gi_dro.npz")
    
    """ Train Yeast Triple GI Model """
    mn_spec = load_spec("cfgs/tgi_mn_model.json")
    train_gi_model(mn_spec, 
                "../generated-data/dataset_yeast_tgi_mn.feather",
                "../generated-data/splits/dataset_yeast_tgi.npz",
                "../results/website_models/tgi_yeast")
    save_model_for_website("../results/website_models/tgi_yeast", "website/models/tgi_yeast.npz")

def train_gi_model(mn_spec, dataset_path, splits_path, trained_model_path):
    df = pd.read_feather(dataset_path)
    d = np.load(splits_path, allow_pickle=True)

    splits = d['splits']
    n_reps = d['reps']
    n_folds = d['folds']

    split_ids = [i * n_folds for i in range(n_reps)]

    return train(mn_spec, df, splits, split_ids, trained_model_path)

def train(model_spec, df, splits, split_ids, output_path):
    os.makedirs(output_path, exist_ok=True)
    model_spec['verbose'] = False 

    futures = []
    model_output_paths = []

    features = models.mn.expand_features(model_spec, df)
    X = df[features].to_numpy()
    _, mu, std = models.common.normalize(X)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(split_ids)) as executor:
        for i, split_id in enumerate(split_ids):
            
            train_df, valid_df, _ = models.common.get_dfs(df, splits[split_id], train_ids=[0, 1, 3], valid_ids=[2], test_ids=[0])
            model_output_path = os.path.join(output_path, 'model_%d.npz' % i)
            model_output_paths.append(model_output_path)
            futures.append(executor.submit(_train_model, model_spec, train_df, valid_df, mu, std, model_output_path))
            #_train_model(model_spec, train_df, valid_df, mu, std, model_output_path)

    concurrent.futures.wait(futures)

    return model_output_paths
    
def _train_model(model_spec, train_df, valid_df, mu, std, output_path):
    model_spec = copy.deepcopy(model_spec)
    
    model = models.mn.MnModel(model_spec)
    
    model.train(train_df, valid_df, mu, std)
    model.save(output_path)

def save_model_for_website(model_output_path, website_model_path):

    files = glob.glob("%s/*.npz" % model_output_path)
    Ws = []
    for file in files:
        print(file)
        model = models.mn.MnModel.load(file)

        biases, coefficients, features = model.get_coefficients(ref_class=1)
        mu, std = model._mu, model._std 

        W = np.hstack((biases[0], coefficients[:, 0]))
        Ws.append(W)
    
    Ws = np.array(Ws)
    
    np.savez(website_model_path, W=Ws.T, 
                                 features=['Bias'] + features,
                                 mu=np.hstack(([0], mu)),
                                 std=np.hstack(([1], std)))

if __name__ == "__main__":
    main()
