import pandas as pd
import numpy as np
import networkx as nx 
import sys 
import scipy.stats
import os 

from utils import yeast_name_resolver

groups = {
    'wt1' : ['WT1'],
    'wt2' : ['WT2'],
    'wt3' : ['WT3'],
    'hu' : ['HU80', 'HU120', 'HU160'],
    'rap' : ['RAP60', 'RAP140', 'RAP220', 'RAP300', 'RAP380', 'RAP460', 'RAP540', 'RAP620', 'RAP700']
}

res = yeast_name_resolver.NameResolver()

def main():
    gpath = sys.argv[1]

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())    
    node_ix = dict(zip(nodes, np.arange(len(nodes))))

    abd_df = read_chong('../data-sources/yeast/chong2015/mmc2.xls')
  
    num_features = len(abd_df.columns) - 1

    F = np.zeros((len(nodes), num_features))
    
    common_nodes = set(nodes) & set(abd_df['gene'])
    print("Common nodes between yeast net and chong dataset: %d" % len(common_nodes))
    abd_df = abd_df.set_index('gene')

    for group in groups:
        target_cols = groups[group]
        F = np.zeros((len(nodes), len(target_cols)))
        for node in common_nodes:
            idx = node_ix[node]
            F[idx, :] = abd_df.loc[node, target_cols]
        #print(scipy.stats.describe(F))

        mu = np.mean(F, axis=0)
        std = np.std(F, axis=0)
        F = scipy.stats.zscore(F, axis=0)
        
        output_path = '../generated-data/features/%s_abundance_%s' % (os.path.basename(gpath), group)

        np.savez(output_path, F=F, mu=mu, std=std, feature_labels=target_cols) 
        #print(scipy.stats.describe(F))
        print(group, " ", F.shape)
        
def read_chong(path):
    df = pd.read_excel(path)
    df = df.fillna(0)

    names = [res.get_unified_name(g) for g in df['ORF']]
    df['gene'] = names
    df = df.drop(['ORF', 'Gene name'], axis=1)
    return df 

if __name__ == "__main__":
    main()