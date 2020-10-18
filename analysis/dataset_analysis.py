import numpy as np 
import pandas as pd 
import sys 
import networkx as nx 

yeast_cfg = {
    "smf_path" : "../generated-data/task_yeast_smf_30",
    "binned_smf_path" : "../generated-data/features/ppc_yeast_smf_binned.npz",
    "gpath" : "../generated-data/ppc_yeast",
    "gi_path" : "../generated-data/task_yeast_gi_hybrid",
    "sgo_path" : "../generated-data/features/ppc_yeast_common_sgo.npz"
}

pombe_cfg = {
    "smf_path" : "../generated-data/task_pombe_smf",
    "binned_smf_path" : "../generated-data/features/ppc_pombe_smf_binned.npz",
    "gpath" : "../generated-data/ppc_pombe",
    "gi_path" : "../generated-data/task_pombe_gi",
    "sgo_path" : "../generated-data/features/ppc_pombe_common_sgo.npz"
}

human_cfg = {
    "smf_path" : "../generated-data/task_human_smf",
    "binned_smf_path" : "../generated-data/features/ppc_human_smf_binned.npz",
    "gpath" : "../generated-data/ppc_human",
    "gi_path" : "../generated-data/task_human_gi",
    "sgo_path" : "../generated-data/features/ppc_human_common_sgo.npz"
}

dro_cfg = {
    "smf_path" : "../generated-data/task_dro_smf",
    "binned_smf_path" : "../generated-data/features/ppc_dro_smf_binned.npz",
    "gpath" : "../generated-data/ppc_dro",
    "gi_path" : "../generated-data/task_dro_gi",
    "sgo_path" : "../generated-data/features/ppc_dro_common_sgo.npz"
}

np.set_printoptions(precision=2)
def main(cfg):

    G = nx.read_gpickle(cfg['gpath'])
    print("Number of genes in network: %d" % len(G.nodes()))

    df = pd.read_csv(cfg['smf_path'])
    bins = sorted(set(df['bin']))
    print()
    print("Size of SMF dataset: %d" % df.shape[0])
    print("Breakdown in bins (raw count): ")
    breakdown = np.array([np.sum(df['bin'] == b) for b in bins])
    print(breakdown)
    print("In percentage:")
    print(breakdown*100/ np.sum(breakdown))

    sgo = np.load(cfg['sgo_path'])
    F = sgo['F']
    ix_no_terms = np.sum(F, axis=1) == 0
    print()
    print("# of genes without sGO terms: %d (%0.2f %% of All Genes)" % (np.sum(ix_no_terms), np.mean(ix_no_terms)*100))
    ix_with_smf = np.zeros(F.shape[0]).astype(bool)
    ix_with_smf[df['id']] = True
    ix = ix_no_terms & ix_with_smf
    n_no_sgo_smf = np.sum(ix)
    print("# of genes without sGO terms but with SMF: %d (%0.2f %% of All Genes)" % (np.sum(ix), np.mean(ix) * 100))
    print("# genes without sGO terms in each bin:")
    for b in bins:
        ix_in_bin = np.zeros(F.shape[0]).astype(bool)
        sdf = df[df['bin'] == b]
        ix_in_bin[sdf['id']] = True

        ix = ix_no_terms & ix_in_bin
        print("Bin %d: %d (Of the bin %0.2f %%) %0.2f %%" % (b, np.sum(ix), np.sum(ix) * 100 / np.sum(df['bin'] == b), np.sum(ix) * 100/ n_no_sgo_smf))
    
    gi_df = pd.read_csv(cfg['gi_path'])
    print()
    print("Size of GI dataset: %d" % gi_df.shape[0])
    print("Breakdown in bins (raw count): ")
    gi_bins = sorted(set(gi_df['bin']))
    breakdown = np.array([np.sum(gi_df['bin'] == b) for b in gi_bins])
    print(breakdown)
    print("In percentage:")
    print(breakdown*100/np.sum(breakdown))

    smf = np.load(cfg['binned_smf_path'])
    F_smf = smf['F']
    a_smf = F_smf[gi_df['a_id'],:]
    b_smf = F_smf[gi_df['b_id'],:]

    ix_a_no_smf = np.sum(a_smf, axis=1) == 0
    ix_b_no_smf = np.sum(b_smf, axis=1) == 0

    ix_no_smf_either = ix_a_no_smf | ix_b_no_smf
    
    
    print()
    print("Pairs with no SMF in at least one of the genes: %d (%0.6f %% of pairs in GI dataset)" % (np.sum(ix_no_smf_either), np.mean(ix_no_smf_either) * 100))
    print("Breakdown of pairs without SMF in the GI dataset:")
    for b in gi_bins:
        ix_b = gi_df['bin'] == b
        ix = ix_b & ix_no_smf_either
        print("Bin %d: %d (of bin %0.2f %%) %0.2f %%" % (b, np.sum(ix), np.sum(ix) * 100/ np.sum(gi_df['bin'] == b), np.sum(ix) * 100 / np.sum(ix_no_smf_either)))


if __name__ == "__main__":

    main(yeast_cfg)
    main(pombe_cfg)
    main(human_cfg)
    main(dro_cfg)