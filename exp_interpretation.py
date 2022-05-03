from asyncio import futures
import json 
import pandas as pd
import numpy as np
import models.mn 
import models.common
import scipy.stats
import os 
import concurrent.futures
import copy
import glob 

with open('../generated-data/go_ids_to_names.json', 'r') as f:
    goid_names = json.load(f)

ORG_NAMES = {
    'yeast' : 'S. cerevisiae',
    'pombe' : 'S. pombe',
    'human' : 'H. sapiens',
    'dro' : 'D. melanogaster'

}
SMF_LABELS = ['Lethal', 'Reduced growth', 'Normal']
SMF_BINARY_LABELS = ['Lethal', 'Viable']
SMF_CA_MO_V = ['Cellular Autonomous', 'Multi Organismal Lethal', 'Viable']
GI_LABELS = ['Negative', 'Neutral', 'Positive', 'Suppression']
BINARY_GI_LABELS = ['Negative', 'Interacting']

def main():

    """ S-MN (3way) """
    smf_mn_spec = load_spec('cfgs/smf_mn_model.json')
    train_models(smf_mn_spec, "../generated-data/dataset_yeast_smf.feather", "../results/exp_interpretation/smf_yeast")
    train_models(smf_mn_spec, "../generated-data/dataset_pombe_smf.feather", "../results/exp_interpretation/smf_pombe")
    train_models(smf_mn_spec, "../generated-data/dataset_human_smf.feather", "../results/exp_interpretation/smf_human")
    train_models(smf_mn_spec, "../generated-data/dataset_dro_smf.feather", "../results/exp_interpretation/smf_dro")
    smf_df = compile_results(['../results/exp_interpretation/smf_%s' % o for o in ['yeast', 'pombe', 'human', 'dro']],
                    [ORG_NAMES[o] for o in ['yeast','pombe','human','dro']],
                    SMF_LABELS, 2)
    
    """ S-MN (binary) """
    smf_mn_spec = load_spec('cfgs/smf_mn_model.json')
    smf_mn_spec['target_col'] = 'is_viable'
    train_models(smf_mn_spec, "../generated-data/dataset_yeast_smf.feather", "../results/exp_interpretation/smf_binary_yeast")
    train_models(smf_mn_spec, "../generated-data/dataset_pombe_smf.feather", "../results/exp_interpretation/smf_binary_pombe")
    train_models(smf_mn_spec, "../generated-data/dataset_human_smf.feather", "../results/exp_interpretation/smf_binary_human")
    train_models(smf_mn_spec, "../generated-data/dataset_dro_smf.feather", "../results/exp_interpretation/smf_binary_dro")
    smf_binary_df = compile_results(['../results/exp_interpretation/smf_binary_%s' % o for o in ['yeast', 'pombe', 'human', 'dro']],
                    [ORG_NAMES[o] for o in ['yeast','pombe','human','dro']],
                    SMF_BINARY_LABELS, 1)

    smf_binary_df.to_excel("../results/exp_interpretation/smf_binary.xlsx")

    """ CA vs MO vs V  """
    smf_mn_spec = load_spec('cfgs/smf_mn_model.json')
    train_models(smf_mn_spec, "../generated-data/dataset_human_smf_ca_mo_v.feather", "../results/exp_interpretation/smf_ca_mo_v_human")
    train_models(smf_mn_spec, "../generated-data/dataset_dro_smf_ca_mo_v.feather", "../results/exp_interpretation/smf_ca_mo_v_dro")
    smf_ca_mo_v_df = compile_results(['../results/exp_interpretation/smf_ca_mo_v_%s' % o for o in ['human', 'dro']],
                    [ORG_NAMES[o] for o in ['human','dro']],
                    SMF_CA_MO_V, 1)

    smf_ca_mo_v_df.to_excel("../results/exp_interpretation/smf_ca_mo_v.xlsx")

    """ D-MN 4way"""
    gi_mn_spec = load_spec('cfgs/gi_mn_model.json')
    train_models(gi_mn_spec, "../generated-data/dataset_yeast_gi_hybrid_mn.feather", "../results/exp_interpretation/gi_yeast", n_epochs=50, n_workers=5)
    train_models(gi_mn_spec, "../generated-data/dataset_pombe_gi_mn.feather", "../results/exp_interpretation/gi_pombe", n_epochs=50, n_workers=5)
    gi_df = compile_results(['../results/exp_interpretation/gi_%s' % o for o in ['yeast', 'pombe']],
                    [ORG_NAMES[o] for o in ['yeast','pombe']],
                    GI_LABELS, 1)

    gi_df.to_excel("../results/exp_interpretation/gi.xlsx")
    
    """ D-MN binary """
    gi_mn_spec = load_spec('cfgs/gi_mn_model.json')
    gi_mn_spec['target_col'] = 'is_neutral'
    train_models(gi_mn_spec, "../generated-data/dataset_yeast_gi_hybrid_mn.feather", "../results/exp_interpretation/gi_binary_yeast", n_epochs=50, n_workers=5)
    train_models(gi_mn_spec, "../generated-data/dataset_pombe_gi_mn.feather", "../results/exp_interpretation/gi_binary_pombe", n_epochs=50, n_workers=5)
    train_models(gi_mn_spec, "../generated-data/dataset_human_gi_mn.feather", "../results/exp_interpretation/gi_binary_human", n_epochs=50, n_workers=5)
    train_models(gi_mn_spec, "../generated-data/dataset_dro_gi_mn.feather", "../results/exp_interpretation/gi_binary_dro", n_epochs=50, n_workers=5)
    gi_binary_df = compile_results(['../results/exp_interpretation/gi_binary_%s' % o for o in ['yeast', 'pombe', 'human', 'dro']],
                    [ORG_NAMES[o] for o in ['yeast','pombe', 'human', 'dro']],
                    BINARY_GI_LABELS, 1)
    gi_binary_df.to_excel("../results/exp_interpretation/gi_binary.xlsx")
def load_spec(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_coefficients(saved_models_path, ref_class):
    files = glob.glob("%s/*.npz" % saved_models_path)

    all_coeff = []
    for file in files:
        model = models.mn.MnModel.load(file)
        biases, coefficients, features = model.get_coefficients(ref_class)

        print(file, " ", np.sum(np.isnan(coefficients)))
        all_coeff.append(coefficients)
    
    muW = np.mean(all_coeff, axis=0)
    stdW = scipy.stats.sem(all_coeff, axis=0)
    

    lower_error = []
    upper_error = []
    for mu, std in zip(muW, stdW):
        lower, upper = scipy.stats.t.interval(0.95, muW.shape[0]-1, loc=mu, scale=std)

        lower_error.append(lower)
        upper_error.append(upper)
    
    lower_error = np.array(lower_error)
    upper_error = np.array(upper_error)

    dfs = []
    features = [process_label(f) for f in features]
    for k in range(muW.shape[1]):
        
        mean_coeff = muW[:, [k]]
        ci_lower = lower_error[:, [k]]
        ci_upper = upper_error[:, [k]]
        exp_mean_coeff = np.exp(mean_coeff)

        data = np.hstack((mean_coeff, ci_lower, ci_upper, exp_mean_coeff))

        df = pd.DataFrame(data=data, index=features, columns=['Mean Coefficient Value', '95% CI Lower', '95% CI Upper', 'Exp Mean Coefficient Value'])
        
        dfs.append(df)
    
    
    return dfs

def train_models(model_spec, dataset_path, output_path, n_resampling=50, n_workers=16, n_epochs=500, batch_size_p=0.1):

    os.makedirs(output_path, exist_ok=True)

    model_spec['epochs'] = n_epochs
    model_spec['early_stopping'] = False 
    model_spec['verbose'] = False 
    model_spec['batch_size_p'] = batch_size_p


    df = pd.read_feather(dataset_path)
    
    futures_not_done = set()
    model_output_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            
        for i in range(n_resampling):
            
            model_output_path = os.path.join(output_path, 'model_%d.npz' % i)
            model_output_paths.append(model_output_path)

            future = executor.submit(_train_model, model_spec, df, model_output_path)
            futures_not_done.add(future)

    concurrent.futures.wait(futures_not_done)
    
    return model_output_paths

def _train_model(model_spec, df, output_path):
    rix = np.random.choice(df.shape[0], df.shape[0], replace=True)
    resampled_df = df.iloc[rix]
            
    model_spec = copy.deepcopy(model_spec)

    model = models.mn.MnModel(model_spec)
    model.train(resampled_df, resampled_df)

    model.save(output_path)

label_lookup = {
    'topology-lid' : 'LID',
    'redundancy-pident' : 'Percent Identity',
}
def process_label(lbl):
    fset, feature = lbl.split('-')

    if fset == 'sgo' and feature in goid_names:
        return goid_names[feature].title()
    
    return label_lookup.get(lbl, lbl)

def compile_results(paths, titles, class_labels, ref_class):

    org_dfs = {}
    for p, path in enumerate(paths):
        dfs = get_coefficients(path , ref_class)
        for r, df in enumerate(dfs):
            df.columns = pd.MultiIndex.from_tuples([("%s (%s)" % (titles[p], class_labels[r]), c) for c in df.columns])
        org_dfs[p] = dfs 
    
    final_df = []
    for c in range(len(class_labels)):
        if c == ref_class:
            continue 
        for org, dfs in org_dfs.items():
            final_df.append(dfs[c])
    final_df = pd.concat(final_df, axis=1)
    
    return final_df

if __name__ == "__main__":
    main()