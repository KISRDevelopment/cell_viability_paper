import numpy as np 
import pandas as pd 

def main():

    compile_dataset("../generated-data/task_yeast_smf_30",
                              [
                                "../generated-data/features/ppc_yeast_topology.npz",
                                "../generated-data/features/ppc_yeast_common_sgo.npz",
                                "../generated-data/features/ppc_yeast_redundancy.npz",
                                "../generated-data/features/ppc_yeast_phosphotase.npz",
                                "../generated-data/features/ppc_yeast_kinase.npz",
                                "../generated-data/features/ppc_yeast_transcription.npz",
                                "../generated-data/features/ppc_yeast_abundance_hu.npz",
                                "../generated-data/features/ppc_yeast_abundance_rap.npz",
                                "../generated-data/features/ppc_yeast_abundance_wt3.npz",
                                "../generated-data/features/ppc_yeast_localization_hu.npz",
                                "../generated-data/features/ppc_yeast_localization_rap.npz",
                                "../generated-data/features/ppc_yeast_localization_wt3.npz"
                            ],
                            [
                                'topology',
                                'sgo',
                                'redundancy',
                                'phosphotase',
                                'kinase',
                                'transcription',
                                'abundance_hu',
                                'abundance_rap',
                                'abundance_wt3',
                                'localization_hu',
                                'localization_rap',
                                'localization_wt3'
                            ], "../generated-data/dataset_yeast_smf")
    
    compile_dataset("../generated-data/task_pombe_smf",
                    [
                        "../generated-data/features/ppc_pombe_topology.npz",
                        "../generated-data/features/ppc_pombe_common_sgo.npz",
                        "../generated-data/features/ppc_pombe_redundancy.npz",
                    ],
                    [
                        "topology",
                        "sgo",
                        "redundancy"
                    ], "../generated-data/dataset_pombe_smf")
    
    compile_dataset("../generated-data/task_human_smf",
                    [
                        "../generated-data/features/ppc_human_topology.npz",
                        "../generated-data/features/ppc_human_common_sgo.npz",
                        "../generated-data/features/ppc_human_redundancy.npz",
                    ],
                    [
                        "topology",
                        "sgo",
                        "redundancy"
                    ], "../generated-data/dataset_human_smf")

    compile_dataset("../generated-data/task_human_smf_ca_mo_v",
                    [
                        "../generated-data/features/ppc_human_topology.npz",
                        "../generated-data/features/ppc_human_common_sgo.npz",
                        "../generated-data/features/ppc_human_redundancy.npz",
                    ],
                    [
                        "topology",
                        "sgo",
                        "redundancy"
                    ], "../generated-data/dataset_human_smf_ca_mo_v")
    
    compile_dataset("../generated-data/task_human_smf_mo_v",
                    [
                        "../generated-data/features/ppc_human_topology.npz",
                        "../generated-data/features/ppc_human_common_sgo.npz",
                        "../generated-data/features/ppc_human_redundancy.npz",
                    ],
                    [
                        "topology",
                        "sgo",
                        "redundancy"
                    ], "../generated-data/dataset_human_smf_mo_v")

    compile_dataset("../generated-data/task_dro_smf",
                    [
                        "../generated-data/features/ppc_dro_topology.npz",
                        "../generated-data/features/ppc_dro_common_sgo.npz",
                        "../generated-data/features/ppc_dro_redundancy.npz",
                    ],
                    [
                        "topology",
                        "sgo",
                        "redundancy"
                    ], "../generated-data/dataset_dro_smf")
    
    compile_dataset("../generated-data/task_dro_smf_ca_mo_v",
                    [
                        "../generated-data/features/ppc_dro_topology.npz",
                        "../generated-data/features/ppc_dro_common_sgo.npz",
                        "../generated-data/features/ppc_dro_redundancy.npz",
                    ],
                    [
                        "topology",
                        "sgo",
                        "redundancy"
                    ], "../generated-data/dataset_dro_smf_ca_mo_v")

    compile_dataset("../generated-data/task_dro_smf_mo_v",
                    [
                        "../generated-data/features/ppc_dro_topology.npz",
                        "../generated-data/features/ppc_dro_common_sgo.npz",
                        "../generated-data/features/ppc_dro_redundancy.npz",
                    ],
                    [
                        "topology",
                        "sgo",
                        "redundancy"
                    ], "../generated-data/dataset_dro_smf_mo_v")

    yeast_features_spec = [
            {
                "path" : "../generated-data/pairwise_features/ppc_yeast_shortest_path_len.npy",
                "name" : "pairwise-spl",
                "reader" : read_dense_pairwise
            },
            {
                "path" : "../generated-data/pairwise_features/ppc_yeast_adhesion.npy",
                "name" : "pairwise-adhesion",
                "reader" : read_dense_pairwise
            },
            {
                "path" : "../generated-data/pairwise_features/ppc_yeast_cohesion.npy",
                "name" : "pairwise-cohesion",
                "reader" : read_dense_pairwise
            },
            {
                "path" : "../generated-data/pairwise_features/ppc_yeast_mutual_neighbors.npy",
                "name" : "pairwise-mutual_neighbors",
                "reader" : read_dense_pairwise
            },
            {
                "path" : "../generated-data/pairwise_features/ppc_yeast_adjacent.npy",
                "name" : "pairwise-adjacent",
                "reader" : read_dense_pairwise
            },
            {
                "path" : "../generated-data/pairwise_features/ppc_yeast_5steps_comms.npz",
                "name" : "pairwise",
                "reader" : read_pairwise_comms
            },
    ]

    compile_gi_dataset("../generated-data/task_yeast_gi_costanzo", 
        yeast_features_spec, 
        "../generated-data/dataset_yeast_gi_costanzo")
    compile_gi_dataset("../generated-data/task_yeast_gi_hybrid", 
        yeast_features_spec, 
        "../generated-data/dataset_yeast_gi_hybrid")
    compile_tgi_dataset("../generated-data/task_yeast_tgi",
        yeast_features_spec,
        "../generated-data/dataset_yeast_tgi"
    )
    other_organisms = ['pombe', 'human', 'dro']
    for org in other_organisms:
        other_org_spec = [
            {
                "path" : "../generated-data/pairwise_features/ppc_%s_shortest_path_len_sparse.npz" % org,
                "name" : "pairwise-spl",
                "reader" : read_sparse_spl
            }
        ]

        compile_gi_dataset("../generated-data/task_%s_gi" % org, 
                          other_org_spec,
                          "../generated-data/dataset_%s_gi" % org)
    
    
def compile_dataset(path, feature_files, feature_sets, output_path):
    print("Compiling ", path)

    df = pd.read_csv(path)
    gene_id = np.array(df['id'])
    
    fs = []
    cols = []
    mus = []
    stds = []
    for feature_file, feature_set in zip(feature_files, feature_sets):
        d = np.load(feature_file)
        F = d['F']

        if len(F.shape) == 2:
            f = F[gene_id,:]
            f_cols = ['%s-%s' % (feature_set,c) for c in d['feature_labels']]

            if 'mu' in d:
                f_mu = d['mu']
                f_std = d['std']
            else:
                f_mu = np.zeros(F.shape[1])
                f_std = np.ones(F.shape[1])
            
            fs.append(f)
            cols.extend(f_cols)
            mus.append(f_mu)
            stds.append(f_std)
        
        elif len(F.shape) == 3:

            for i in range(F.shape[1]):
                sub_F = F[:, i, :]
                f = sub_F[gene_id,:]
                f_cols = ['%s-comp%d_%s' % (feature_set, i,c) for c in d['feature_labels']]
                f_mu = d['mu'][i,:]
                f_std = d['std'][i,:]

                fs.append(f)
                cols.extend(f_cols)
                mus.append(f_mu)
                stds.append(f_std)
    
    mu = np.hstack(mus)
    std = np.hstack(stds)

    F = np.hstack(fs) * std + mu 
    assert np.sum(np.isnan(F)) == 0
    
    F_df = pd.DataFrame(data=F, columns=cols)
    
    df = pd.concat((df, F_df), axis=1)
    df.to_csv(output_path, index=False)
    print(df.shape)

def compile_gi_dataset(path, spec, output_path):
    print("Compiling ", path)

    df = pd.read_csv(path)

    a_id = np.array(df['a_id'])
    b_id = np.array(df['b_id'])

    Fs = []
    cols = []

    for s in spec:

        func = s['reader']

        f, f_cols = func(a_id, b_id, s['path'])

        Fs.append(f)

        if f_cols is None:
            f_cols = [s['name']]
        else:
            f_cols = ['%s-%s' % (s['name'], c) for c in f_cols]
        
        cols.extend(f_cols)
    
    F_df = pd.DataFrame(data=np.hstack(Fs), columns=cols)

    df = pd.concat([df, F_df], axis=1)

    df.to_csv(output_path, index=False)
    print(df.shape)
def compile_tgi_dataset(path, spec, output_path):
    print("Compiling ", path)

    df = pd.read_csv(path)

    a_id = np.array(df['a_id'])
    b_id = np.array(df['b_id'])
    c_id = np.array(df['c_id'])
    
    Fs = []
    cols = []

    for s in spec:

        func = s['reader']

        fab, fab_cols = func(a_id, b_id, s['path'])
        fac, fac_cols = func(a_id, c_id, s['path'])
        fbc, fbc_cols = func(b_id, c_id, s['path'])
        
        Fs.append(fab)
        Fs.append(fac)
        Fs.append(fbc)
        cols.extend(expand_col_names('ab-%s' % s['name'], fab_cols))
        cols.extend(expand_col_names('ac-%s' % s['name'], fac_cols))
        cols.extend(expand_col_names('bc-%s' % s['name'], fbc_cols))
    
    F_df = pd.DataFrame(data=np.hstack(Fs), columns=cols)

    df = pd.concat([df, F_df], axis=1)

    df.to_csv(output_path, index=False)
    print(df.shape)
def expand_col_names(name, f_cols):
    if f_cols is None:
        f_cols = [name]
    else:
        f_cols = ['%s-%s' % (name, c) for c in f_cols]
    return f_cols 

def read_dense_pairwise(a_id, b_id, path):
    F = np.load(path)
    return F[a_id, b_id, None], None

def read_sparse_spl(a_id, b_id, path):
    d = np.load(path, allow_pickle=True)
    
    node_id_to_comp = d['node_id_to_comp']
    Ps = d['Ps']

    F = np.zeros((a_id.shape[0], 1))

    # things that belong to diff components have _infinite_ distance 
    ix_same_comp = node_id_to_comp[a_id, 0] == node_id_to_comp[b_id, 0]
    F[~ix_same_comp] = 1e5

    # get things that belong to same components
    ix_same_comp = np.where(ix_same_comp)[0]
    for idx in ix_same_comp:
        # get the node ids
        a, b = a_id[idx], b_id[idx]

        assert node_id_to_comp[a,0] == node_id_to_comp[b,0]

        # get the corresponding distance matrix
        P = Ps[node_id_to_comp[a,0]]

        # get their index within the distance matrix
        a_idx = node_id_to_comp[a,1]
        b_idx = node_id_to_comp[b,1]

        # set value
        F[idx] = P[a_idx][b_idx]

        assert not np.isnan(F[idx])
    
    return F, None

def read_pairwise_comms(a_id, b_id, path):
    d = np.load(path, allow_pickle=True)
    
    indecies = d['indecies']
    data = d['data']

    df_indecies = [tuple(sorted((a,b))) for a,b in zip(a_id, b_id)]
    
    indecies_to_data = { tuple(k):v for k,v in zip(indecies, data) }
    
    F = [indecies_to_data.get(dfi, [0,0,0]) for dfi in df_indecies]
    
    return np.array(F), ['within_comm', 'cross_comm', 'same_comm']



if __name__ == "__main__":
    main()
