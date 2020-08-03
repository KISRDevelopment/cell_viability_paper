import os
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import sys
import networkx as nx
from collections import defaultdict
import json 

STANDARDIZE = True 

def main(gpath, smf_task_path):
    G = nx.read_gpickle(gpath)
    nodes = list(sorted(G.nodes()))
    
    smf_df = pd.read_csv(smf_task_path)
    smf_df = smf_df.set_index('gene')

    available_smf = set(smf_df.index)

    F = np.zeros((len(nodes), 2))
    for node_id, node in enumerate(nodes):
        
        if node not in available_smf:
            continue 
        
        row = smf_df.loc[node]

        # smf, essential or not
        F[node_id, 0] = row['cs']
        F[node_id, 1] = int(row['bin'] == 0)

    path = '../generated-data/features/%s_smf' % (os.path.basename(gpath))

    if STANDARDIZE:
        F[:, 0] = stats.zscore(F[:, 0])
        
    print(stats.describe(F))
    
    print("Writing to %s" % path)
    np.savez(path, 
        F=F, feature_labels=['smf', 'essential'])


if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath)
    
