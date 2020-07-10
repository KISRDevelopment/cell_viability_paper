import numpy as np
import pandas as pd
import sys
import os 
import scipy.stats as stats

def main():
    path = sys.argv[1]
    
    df = pd.read_csv(path)
    
    bins = np.array(df['bin'])
    
    bin_counts = [np.sum(bins == b) for b in sorted(np.unique(bins))]
    print(bin_counts)
    print(np.array(bin_counts) * 100 / np.sum(bin_counts))

    output_path = "../generated-data/targets/%s_bin_simple" % (os.path.basename(path))
    np.savez(output_path, y=bins)

if __name__ == "__main__":
    main()