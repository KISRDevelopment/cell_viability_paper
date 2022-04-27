import numpy as np 
import pandas as pd 
import feature_preprocessing.mn_features

def main():
    compile_gi_mn_dataset("../generated-data/dataset_yeast_gi_costanzo.feather",
                         "../generated-data/dataset_yeast_allppc.feather",
                         "../generated-data/dataset_yeast_gi_costanzo_mn.feather")
    compile_gi_mn_dataset("../generated-data/dataset_yeast_gi_hybrid.feather",
                         "../generated-data/dataset_yeast_allppc.feather",
                         "../generated-data/dataset_yeast_gi_hybrid_mn.feather")
    compile_tgi_mn_dataset("../generated-data/dataset_yeast_tgi.feather",
                         "../generated-data/dataset_yeast_allppc.feather",
                         "../generated-data/dataset_yeast_tgi_mn.feather")
    compile_tgi_mn_dataset("../generated-data/dataset_yeast_pseudo_triplets.feather",
                         "../generated-data/dataset_yeast_allppc.feather",
                         "../generated-data/dataset_yeast_pseudo_triplets_mn.feather")
    compile_gi_mn_dataset("../generated-data/dataset_pombe_gi.feather",
                         "../generated-data/dataset_pombe_smf.feather",
                         "../generated-data/dataset_pombe_gi_mn.feather")
    compile_gi_mn_dataset("../generated-data/dataset_human_gi.feather",
                         "../generated-data/dataset_human_smf.feather",
                         "../generated-data/dataset_human_gi_mn.feather")
    compile_gi_mn_dataset("../generated-data/dataset_dro_gi.feather",
                         "../generated-data/dataset_dro_smf.feather",
                         "../generated-data/dataset_dro_gi_mn.feather")
    
def compile_gi_mn_dataset(path, smf_path, output_path):

    spec = [
        "pairwise-spl",
        { "op" : "add", "feature" : "topology-lid" },
        { "op" : "combs", "feature" : "bin" },
        { "op" : "add", "feature" : "sgo-" }
    ]

    smf_df = pd.read_feather(smf_path)
    gi_df = pd.read_feather(path)

    feature_preprocessing.mn_features.create_double_gene_mn_features(spec, smf_df, gi_df, output_path)

def compile_tgi_mn_dataset(path, smf_path, output_path):

    spec = [
        { "op" : "add", "feature" : "sgo-", "type" : "single" },
        { "op" : "add", "feature" : "topology-lid", "type" : "single" },
        { "op" : "add", "feature" : "pairwise-spl", "type" : "pair" },
        { "op" : "combs", "feature" : "bin", "type" : "single" }
    ]

    smf_df = pd.read_feather(smf_path)
    tgi_df = pd.read_feather(path)

    feature_preprocessing.mn_features.create_triple_gene_mn_features(spec, smf_df, tgi_df, output_path)

if __name__ == "__main__":
    main()
