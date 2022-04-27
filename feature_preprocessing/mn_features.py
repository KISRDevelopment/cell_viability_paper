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

SMF_BIN_LABELS = np.array(['L', 'R', 'N'])

def create_triple_gene_mn_features(spec, smf_df, gi_df, output_path):
    smf_df_cols = smf_df.columns
    gi_df_cols = gi_df.columns

    a_id = gi_df['a_id']
    b_id = gi_df['b_id']
    c_id = gi_df['c_id']

    smf_df = smf_df.set_index('id')

    dfs = [gi_df[['a_id', 'b_id', 'c_id', 'bin']]]
    for feature in spec:
        
        # direct column feature from the gi_df
        if type(feature) == str:
            cols = gi_df_cols[ gi_df_cols.str.startswith(feature) ]
            sdf = gi_df[cols]
            dfs.append(sdf)
        
        # pairwise feature based on single gene features
        elif type(feature) == dict and feature['type'] == 'single':

            cols = smf_df_cols[ smf_df_cols.str.startswith(feature['feature']) ]
            
            a_features = np.array(smf_df.loc[a_id][cols])
            b_features = np.array(smf_df.loc[b_id][cols])
            c_features = np.array(smf_df.loc[c_id][cols])

            if feature['op'] == 'add':
                added_features = a_features + b_features + c_features
                sdf = pd.DataFrame(data=added_features, index=gi_df.index, columns=cols)
                dfs.append(sdf)
            elif feature['op'] == 'combs':  
                tgi_smf = np.hstack((a_features, b_features, c_features))
                nan_ix = np.isnan(np.sum(tgi_smf, axis=1))
                tgi_smf = tgi_smf[~nan_ix,:]
                tgi_smf = np.sort(tgi_smf, axis=1)
                unique_combs, r_index = np.unique(tgi_smf, axis=0, return_inverse=True)
                n_combs = unique_combs.shape[0]
                unique_combs = unique_combs.astype(int)
                cat_bins = np.zeros((gi_df.shape[0], n_combs))
                cat_bins[~nan_ix, r_index] = 1    

                colnames = ['smf-%s' % (''.join(r)) for r in SMF_BIN_LABELS[unique_combs]]
                sdf = pd.DataFrame(data=cat_bins, index=gi_df.index, columns=colnames)
                dfs.append(sdf)

        elif type(feature) == dict and feature['type'] == 'pair':

            ab_features = np.array(gi_df[ gi_df_cols[gi_df_cols.str.startswith('ab-%s' % feature['feature'])] ])
            ac_features = np.array(gi_df[ gi_df_cols[gi_df_cols.str.startswith('ac-%s' % feature['feature'])] ])
            bc_features = np.array(gi_df[ gi_df_cols[gi_df_cols.str.startswith('bc-%s' % feature['feature'])] ])
            
            if feature['op'] == 'add':
                added_features = ab_features + ac_features + bc_features
                
                colnames = gi_df_cols[gi_df_cols.str.startswith('ab-%s' % feature['feature'])]
                colnames = colnames.str.replace('ab-','abc-')
                sdf = pd.DataFrame(data=added_features, index=gi_df.index, columns=colnames)
                dfs.append(sdf)
            else:
                raise Exception("Unknown op")
        
    final_df = pd.concat(dfs, axis=1)
    
    final_df.to_feather(output_path)
    print(final_df)

if __name__ == "__main__":
    import sys 

    # spec = [
    #     "pairwise-spl",
    #     { "op" : "add", "feature" : "topology-lid" },
    #     { "op" : "combs", "feature" : "bin" },
    #     { "op" : "add", "feature" : "sgo-" }
    # ]

    # smf_df = pd.read_feather("../generated-data/dataset_yeast_allppc.feather")
    # gi_df = pd.read_feather("../generated-data/dataset_yeast_gi_hybrid.feather")

    # create_double_gene_mn_features(spec, smf_df, gi_df, "../tmp/mnfeatures.feather")

    spec = [
        { "op" : "add", "feature" : "sgo-", "type" : "single" },
        { "op" : "add", "feature" : "topology-lid", "type" : "single" },
        { "op" : "add", "feature" : "pairwise-spl", "type" : "pair" },
        { "op" : "combs", "feature" : "bin", "type" : "single" }
    ]
    smf_df = pd.read_feather("../generated-data/dataset_yeast_allppc.feather")
    gi_df = pd.read_feather("../generated-data/dataset_yeast_tgi.feather")
    create_triple_gene_mn_features(spec, smf_df, gi_df, "../tmp/tgi_mn.feather")

    gi_df = pd.read_feather("../generated-data/dataset_yeast_pseudo_triplets.feather")
    create_triple_gene_mn_features(spec, smf_df, gi_df, "../tmp/pseudo_triplets_mn.feather")
