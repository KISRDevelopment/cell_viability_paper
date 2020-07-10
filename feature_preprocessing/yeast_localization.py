import pandas as pd 
import numpy as np 
import networkx as nx 
import sys 
import os 
import scipy.stats as stats

from utils import yeast_name_resolver

res = yeast_name_resolver.NameResolver()

temporal_conditions = {
    'rap' : ['RAP60', 'RAP140', 'RAP220', 'RAP300', 'RAP380', 'RAP460', 'RAP540', 'RAP620', 'RAP700'],
    'hu' : ['HU80', 'HU120', 'HU160'],
    'wt1' : ['WT1'],
    'wt2' : ['WT2'],
    'wt3' : ['WT3']
}

def main(gpath):
    
    G = nx.read_gpickle(gpath)

    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    df = pd.ExcelFile('../data-sources/yeast/chong2015/mmc3.xlsx')

    compartments = get_compartments(df)

    # for every condition, we will have a feature set
    # (N x Compartments x time steps)
    written_files = []
    for condition, sheet_names in temporal_conditions.items():
        
        F = np.zeros((len(node_ix), len(compartments), len(sheet_names)))

        for i, sheet_name in enumerate(sheet_names):

            
            s = df.parse(sheet_name)

            assert list(s.columns[2:]) == compartments

            names = [res.get_unified_name(g) for g in s['ORF']]
            localization = np.array(s[s.columns[2:]])

            for j, name in enumerate(names):
                if name not in node_ix:
                    continue 
                
                nid = node_ix[name]
                F[nid, :, i] = localization[j,:]
        
        mu = np.mean(F, axis=0)
        std = np.std(F, axis=0)
        #F = stats.zscore(F, axis=0)
        #print(stats.describe(F))

        print("Condition %s: %s" % (condition, F.shape))
        
        output_path = '../generated-data/features/%s_localization_%s' % (os.path.basename(gpath), condition)
        np.savez(output_path, F=F, feature_labels=sheet_names, 
            mu=mu, std=std)

        

def get_compartments(df):
    s = df.parse(df.sheet_names[0])
    return list(s.columns[2:])

if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath)
