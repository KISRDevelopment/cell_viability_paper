import numpy as np 
import pandas as pd 
import sys 

def main(path, quality_level):

    df = pd.read_csv(path, sep='\t', header=0, names=['a', 'b', 'official_a', 'official_b', 'organism', 'source', 'ref', 'quality'])
    
    ix_positives = df['source'].str.contains('BioGRID') & (df['quality'] >= quality_level)
    ix_pp = df['source'].str.contains('Slorth')
    ix_tp = ix_positives & ix_pp

    print("Positives in BioGRID: %d" % (np.sum(ix_positives)))
    print("Predicted positive: %d" % np.sum(ix_pp))
    print("Predicted positive & BioGRID: %d" % (np.sum(ix_tp)))
    print("TPR: %0.2f" % (np.sum(ix_tp) / np.sum(ix_positives)))

if __name__ == "__main__":
    path = sys.argv[1]
    quality_level = int(sys.argv[2])
    main(path, quality_level)
