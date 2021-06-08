import numpy as np 
import pandas as pd 

FEATURES_PATH = "/home/mmkhajah/Downloads/slant_data_dir/data_dir/training/yeast_full_features.csv"
SL_PATH = "/home/mmkhajah/Downloads/slant_data_dir/data_dir/sl_files/processed/yeast_sl.csv"

def main():

    sl_df = pd.read_csv(SL_PATH, sep=',')
    ix = sl_df['sl'] == 1
    sl_df = sl_df[ix]
    
    sl_pairs = set([tuple(sorted(t)) for t in zip(sl_df['gene1'], sl_df['gene2'])])
    print(len(sl_pairs))
    

    full_df = pd.read_csv(FEATURES_PATH, sep='\t')
    print(full_df.columns)
    print(full_df.shape)

    full_pairs = [tuple(sorted(t)) for t in zip(full_df['gene1'], full_df['gene2'])]
    full_df['sl'] = [p in sl_pairs for p in full_pairs]

    print(np.sum(full_df['sl']))

if __name__ == "__main__":
    main()
