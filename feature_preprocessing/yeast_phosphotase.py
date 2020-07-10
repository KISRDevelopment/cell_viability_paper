import numpy as np
import os 
import re 
import networkx as nx 
import scipy.stats as stats
import sys 

from utils import yeast_name_resolver

res = yeast_name_resolver.NameResolver()

def main(gpath, file_path):
    
    
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, np.arange(len(nodes))))

    relations = []
    with open(file_path, 'r') as f:
        for line in f:
            m = re.match(r'^(.+?)\s', line)
            if not m:
                continue   
            source = res.get_unified_name(m.group(1))
            m = re.search(r'\[(.+?)\]', line)
            if not m:
                continue 
            targets = res.get_unified_names([t.lower() for t in m.group(1).split(', ')])
            for target in targets:
                relations.append((source,target))
    
    #print(relations)
    print("# relations: %d" % len(relations))
    relations = [rel for rel in relations if rel[0] in node_ix and rel[1] in node_ix]
    print("# relations in network: %d" % len(relations))

    # node has in and out
    F = np.zeros((len(nodes), 2))
    
    for rel in relations:
        src, tar = rel 
        
        src_idx = node_ix[src]
        tar_idx = node_ix[tar]

        # increment in degree
        F[tar_idx, 0] += 1

        # increment out degree 
        F[src_idx, 1] += 1

    ix = (F[:,0] > 0)
    print("Avg in degree: %f" % np.mean(F[ix,0]))
    ix = (F[:,1] > 0)
    print("Avg out degree: %f" % np.mean(F[ix,1]))

    mu = np.mean(F, axis=0)
    std = np.mean(F, axis=0)
    print(stats.describe(F))
    F = stats.zscore(F, axis=0)
    
    print("# genes: %d" % np.sum((F[:,0] > 0) + (F[:,1] > 0)))

    file_path_base =  os.path.basename(file_path).replace('.txt','')
    output_path = '../generated-data/features/%s_%s' % (os.path.basename(gpath), file_path_base)
    np.savez(output_path, F=F, feature_labels=['%s_in_degree' % file_path_base, '%s_out_degree' % file_path_base],
        mu=mu, std=std)
    
if __name__ == "__main__": 
    gpath = sys.argv[1]
    file_path = sys.argv[2]

    main(gpath, file_path)