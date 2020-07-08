import pandas as pd
import numpy as np
import json
import networkx as nx
import sys
import scipy.stats as stats
import os 

from utils import yeast_name_resolver

res = yeast_name_resolver.NameResolver()

TF_THRESHOLD = 1.0


def main():
    gpath = sys.argv[1]
    

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))

    
    F25, labels25, mu25, std25 = read_file(node_ix, '../data-sources/yeast/transcript_25c.xlsx')
    F30, labels30, mu30, std30 = read_file(node_ix, '../data-sources/yeast/transcript_30c.xlsx')
    
    F = np.hstack((F25, F30))
    labels = labels25 + labels30
    mu = np.hstack((mu25,mu30))
    std = np.hstack((std25,std30))

    output_path = '../generated-data/features/%s_transcription' % (os.path.basename(gpath))
    print("Writing to %s" % output_path)

    print(stats.describe(F))
    print(labels)
    print(mu)
    print(std)

    np.savez(output_path, 
        F=F, feature_labels=labels, mu=mu, std=std)
    
def read_file(node_ix, filepath):
    
    df = pd.read_excel(filepath)
    
    factors = [res.get_unified_name(f) for f in df.columns[2:]]
    
    all_columns = np.array(df.columns[:])
    all_columns[2:] = factors
    df.columns = all_columns

    factors = [f for f in factors if f in node_ix]

    # read transcription matrix
    # 0 = In, 1 = Out
    F = np.zeros((len(node_ix), 2))
    for i, r in df.iterrows():
        utarget = res.get_unified_name(r['Factor'])
        if utarget not in node_ix:
            continue
        
        tar_ix = node_ix[utarget]
        for factor in factors:
            
            if r[factor] >= TF_THRESHOLD:
                
                src_ix = node_ix[factor]

                # increment in degree
                F[tar_ix,0] += 1

                # increment out degree
                F[src_ix,1] += 1
    
    ix = (F[:,0] > 0)
    print("Avg in degree: %f" % np.mean(F[ix,0]))
    ix = (F[:,1] > 0)
    print("Avg out degree: %f" % np.mean(F[ix,1]))

    mu = np.mean(F, axis=0)
    std = np.std(F, axis=0)
    F = stats.zscore(F, axis=0)
    
    print("# genes: %d" % np.sum((F[:,0] > 0) + (F[:,1] > 0)))

    filepath_base = os.path.basename(filepath).replace('.xlsx','')

    return F, ['%s_in_degree' % filepath_base, '%s_out_degree' % filepath_base], mu, std

if __name__ == "__main__":
    main()