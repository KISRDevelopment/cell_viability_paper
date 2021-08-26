import numpy as np
import pandas as pd
import sys
import os 
import scipy.stats as stats

def main(input_path, extra_cols, output_path, bin_col='bin'):

    df = pd.read_csv(input_path)
    bins = np.array(df[bin_col])
    
    bin_counts = [np.sum(bins == b) for b in sorted(np.unique(bins))]
    print(bin_counts)
    print(np.array(bin_counts) * 100 / np.sum(bin_counts))

    for col, colfunc in extra_cols.items():
        df[col] = colfunc(bins).astype(int)

    print(np.sum(df[extra_cols.keys()]))
    
    df.to_csv(output_path, index=False)
    print(df)
    # output_path = "../generated-data/targets/%s_%s_simple" % (os.path.basename(path), bin_col)
    # np.savez(output_path, y=bins)
    
    # return output_path+'.npz' 

if __name__ == "__main__":
    main(sys.argv[1])