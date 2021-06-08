import numpy as np
import pandas as pd
import sys
import os 
import scipy.stats as stats

def main(path, bin_col='bin'):
    
    df = pd.read_csv(path)

    
    bins = np.array(df[bin_col])
    neg_ix = bins == 0

    bins[neg_ix] = 0
    bins[~neg_ix] = 1

    bin_counts = [np.sum(bins == b) for b in sorted(np.unique(bins))]
    print(bin_counts)
    print(np.array(bin_counts) * 100 / np.sum(bin_counts))

    output_path = "../generated-data/targets/%s_%s_negative" % (os.path.basename(path), bin_col)
    np.savez(output_path, y=bins)

if __name__ == "__main__":
    main(sys.argv[1])