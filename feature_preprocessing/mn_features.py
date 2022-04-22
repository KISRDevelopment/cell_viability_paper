import numpy as np 
import pandas as pd 
import json 
import tensorflow.keras.utils  as utils

GI_SMF_MAP = np.array([0, 1, 2, 1, 3, 4, 2, 4, 5, 6])
GI_SMF_LABELS = ['LL', 'LR', 'LN', 'RR', 'RN', 'NN']

def create_double_gene_mn_features(spec, smf_df, gi_df, output_path):
    smf_df_cols = smf_df.columns
    gi_df_cols = gi_df.columns

    a_id = gi_df['a_id']
    b_id = gi_df['b_id']

    smf_df = smf_df.set_index('id')

    dfs = [gi_df[['a_id', 'b_id', 'bin']]]
    for feature in spec:
        
        # direct column feature from the gi_df
        if type(feature) == str:
            cols = gi_df_cols[ gi_df_cols.str.startswith(feature) ]
            sdf = gi_df[cols]
            dfs.append(sdf)
        
        # pairwise feature based on single gene features
        elif type(feature) == dict:

            cols = smf_df_cols[ smf_df_cols.str.startswith(feature['feature']) ]
            
            a_features = np.array(smf_df.loc[a_id][cols])
            b_features = np.array(smf_df.loc[b_id][cols])

            if feature['op'] == 'add':
                added_features = a_features + b_features
                sdf = pd.DataFrame(data=added_features, index=gi_df.index, columns=cols)
                dfs.append(sdf)
            elif feature['op'] == 'combs':
                
                # convert combinations to flat index
                eff_bins = (3 * a_features + b_features).squeeze()

                # identify pairs with one or both genes have no SMF
                nan_ix = np.isnan(eff_bins)

                # set those to a unique index
                eff_bins[nan_ix] = 9
                eff_bins = GI_SMF_MAP[eff_bins.astype(int)]
                
                # only set bits for pairs where both genes have smf
                cat_bins = np.zeros((gi_df.shape[0], 6))
                cat_bins[~nan_ix, eff_bins[~nan_ix]] = 1
                
                sdf = pd.DataFrame(data=cat_bins, index=gi_df.index, columns=['smf-%s' % l for l in GI_SMF_LABELS])
                dfs.append(sdf)
    
    final_df = pd.concat(dfs, axis=1)
    
    final_df.to_feather(output_path)
    print(final_df)

if __name__ == "__main__":
    import sys 

    spec = [
        "pairwise-spl",
        { "op" : "add", "feature" : "topology-lid" },
        { "op" : "combs", "feature" : "bin" },
        { "op" : "add", "feature" : "sgo-" }
    ]

    smf_df = pd.read_feather("../generated-data/dataset_yeast_allppc.feather")
    gi_df = pd.read_feather("../generated-data/dataset_yeast_gi_hybrid.feather")

    create_double_gene_mn_features(spec, smf_df, gi_df, "../tmp/mnfeatures.feather")