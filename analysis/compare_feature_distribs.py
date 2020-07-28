import numpy as np 
import pandas as pd 
import scipy.stats
import sys 
import os 
def main(files, fid, remove_zeros=False):
    
    vals = []
    for file in files:
        d = np.load(file)
        labels = d['feature_labels'].tolist()
        f = d['F'][:, fid]
        f = f * d['std'][fid] + d['mu'][fid]
        
        if remove_zeros:
            ix = f > 0
            f = f[ix]
        vals.append(f)
        print("%s: %0.2f" % (os.path.basename(file), np.median(f)))

    print("KRUSKAL TEST:")
    r = scipy.stats.kruskal(*vals)
    print(r)
    print()

    print("PAIRWISE COMPARISONS:")
    n_comparisons = 0
    for i in range(len(files)):
        a = vals[i]
        for j in range(i+1, len(files)):
            b = vals[j]
            r = scipy.stats.kruskal(a, b)
            print("%s AND %s" % (os.path.basename(files[i]), os.path.basename(files[j])))
            print(r)
            print()
            n_comparisons += 1
    
    print("Adjusted alpha: %0.6f" % (0.05 / n_comparisons))


if __name__ == "__main__":
    
    fid = int(sys.argv[1])
    files = sys.argv[2:]

    main(files, fid, remove_zeros=True)
