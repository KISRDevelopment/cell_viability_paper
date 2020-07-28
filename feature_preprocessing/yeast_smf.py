import os
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import sys
import networkx as nx
from collections import defaultdict
import json 

INPUT_PATH = '../generated-data/costanzo_gi'

STANDARDIZE = True 

def main(gpath):
    
    G = nx.read_gpickle(gpath)
    node_set = set(G.nodes())
    nodes =  sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    smf_by_gene = defaultdict(lambda: { 26: [], 30: [] })
    essentials = set()

    df = pd.read_csv(INPUT_PATH)

    rows_a_smf = list(df['a_smf'])
    rows_b_smf = list(df['b_smf'])
    rows_a_essential = list(df['a_essential'])
    rows_b_essential = list(df['b_essential'])
    rows_a = list(df['a'])
    rows_b = list(df['b'])
    rows_temp = list(df['temp'])
    
    for i in range(df.shape[0]):
        
        if rows_a_essential[i] == 1:
            essentials.add(rows_a[i])
        
        if rows_b_essential[i] == 1:
            essentials.add(rows_b[i])
        
        T = rows_temp[i]

        smf_by_gene[rows_a[i]][T].append(rows_a_smf[i])
        smf_by_gene[rows_b[i]][T].append(rows_b_smf[i])
    
    print("# of nodes with data: %d" % len(smf_by_gene))

    # smf @ 26, smf @ 30, essential or not
    F = np.zeros((len(nodes), 3))

    cnt = 0
    applicable_nodes = 0
    for node in smf_by_gene:
        if node not in node_ix:
            continue
        applicable_nodes += 1
        nid = node_ix[node]
        F[nid, 0] = np.nanmean(smf_by_gene[node][26])
        F[nid, 1] = np.nanmean(smf_by_gene[node][30])
        F[nid, 2] = int(node in essentials)
        if F[nid,2] == 0:
            if not(F[nid, 0] > 0 or F[nid, 1] > 0):
                print("Failure at %s: marked nonessential but has no smf at 26 or 30" % node)
                cnt += 1
    print("Failures: %d" % cnt)
    print("Nodes in network with  data: %d" % applicable_nodes)
    F = np.nan_to_num(F)
    print(F.shape)

    output_path = '../generated-data/features/%s_smf' % (os.path.basename(gpath))
    if STANDARDIZE:
        F[:, 0] = stats.zscore(F[:, 0])
        F[:, 1] = stats.zscore(F[:, 1])
        
    np.savez(output_path, F=F, feature_labels=['smf26', 'smf30', 'essential'])


if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath)
    
