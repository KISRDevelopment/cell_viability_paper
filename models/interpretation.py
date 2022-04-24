import json 
import pandas as pd
import numpy as np
import models.mn 
import models.common
import scipy.stats
def train(model_spec, n_models, train_df, output_path):
    model_spec['early_stopping'] = False 

    for i in range(n_models):
        rix = np.random.choice(train_df.shape[0], train_df.shape[0], replace=True)
        resampled_df = train_df.iloc[rix]

        model = models.mn.MnModel(model_spec)
        model.train(resampled_df, resampled_df)
        model.save("%s_%d.npz" % (output_path, i))

def get_coefficients(n_models, output_path, ref_class):
    all_coeff = []
    for i in range(n_models):
        model = models.mn.MnModel.load("%s_%d.npz" % (output_path, i))
        biases, coefficients, features = model.get_coefficients(ref_class)
        all_coeff.append(coefficients)
    mean_coeff = np.mean(all_coeff, axis=0)
    sem_coeff = scipy.stats.sem(all_coeff, axis=0)

    df_mu = pd.DataFrame(data=mean_coeff, index=features)
    df_std = pd.DataFrame(data=sem_coeff, index=features)
    print(df_std)

def main():

    with open('cfgs/smf_mn_model.json', 'r') as f:
        model_spec = json.load(f)
    model_spec['epochs'] = 500

    df = pd.read_feather('../generated-data/dataset_yeast_smf.feather')

    splits = np.load("../generated-data/splits/task_yeast_smf_30.npz")['splits']

    train_df, _, _ = models.common.get_dfs(df, splits[0], train_ids=[0, 1, 2, 3], valid_ids=[2], test_ids=[0])

    #train(model_spec, 10, train_df, "../tmp/yeast_mn_model_interpretation")

    get_coefficients(10, "../tmp/yeast_mn_model_interpretation", 0)
    

if __name__ == "__main__":
    main()