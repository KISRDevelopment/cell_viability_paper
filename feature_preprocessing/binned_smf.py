import numpy as np 
import networkx as nx 
import sys 
import pandas as pd
import os 

LABELS = ['L', 'R', 'N']

def main(gpath, smf_task_path):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    df = pd.read_csv(smf_task_path)

    bins = [LABELS[int(i)] for i in sorted(set(df['bin']))]

    F = np.zeros((len(nodes), len(bins)))

    df_id = np.array(df['id'])
    df_bin = np.array(df['bin']).astype(int)

    for i in range(df_id.shape[0]):
        F[df_id[i], df_bin[i]] = 1
    
    output_path = "../generated-data/features/%s_smf_binned" % (os.path.basename(gpath))
    
    print(np.sum(F, axis=0))
    np.savez(output_path, F=F, feature_labels=bins)
    print("Wrote to %s" % output_path)

if __name__ == "__main__":
    gpath = sys.argv[1]
    smf_task_path = sys.argv[2]

    main(gpath, smf_task_path)