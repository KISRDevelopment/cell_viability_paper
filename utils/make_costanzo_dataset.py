#
# This tool consumes costanzo's dataset and writes it out in a
# format that makes it easy to generate task files from it.
#
# It also filters entries where either gene in the pair is essential
# yet not temp sensetive allele, and vice versa.
#

import os
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import sys
from collections import defaultdict
import json 

import utils.yeast_name_resolver

nr = utils.yeast_name_resolver.NameResolver()

PATHS = ['../data-sources/yeast/costanzo2016/SGA_ExE.txt', # should always come first
         '../data-sources/yeast/costanzo2016/SGA_ExN_NxE.txt',
         '../data-sources/yeast/costanzo2016/SGA_NxN.txt'
]

def main():
    
    essentials = set()
    rows = []
    smf_by_gene = defaultdict(lambda: { 26: [], 30: [] })
    
    for pid, path in enumerate(PATHS):
        print("Reading %s" % path)

        df = pd.read_csv(path, sep='\t')
        
        rows_queries, rows_qtypes = get_queries_and_types(df)
        rows_arrays, rows_atypes, rows_temp = get_arrays_and_types(df)
        rows_queries_smf = list(df['Query single mutant fitness (SMF)'])
        rows_arrays_smf = list(df['Array SMF'])
        rows_cs = list(df['Double mutant fitness'])
        rows_std = list(df['Double mutant fitness standard deviation'])
        rows_gi = list(df['Genetic interaction score (ε)'])
        rows_p = list(df['P-value'])

        print(set(rows_qtypes))
        print(set(rows_atypes))

        
        if 'ExE' in path:
            essentials = set(rows_queries).union(rows_arrays)
        
        skipped = [0,0,0,0]
        cnt_nn = 0
        for i in range(df.shape[0]):
            q = rows_queries[i]
            a = rows_arrays[i]

            if rows_qtypes[i] == 's':
                continue 
            
            if q in essentials:
                if rows_qtypes[i] == 'sn':
                    skipped[0] += 1
                    continue 
            else:
                if rows_qtypes[i] == 'tsq':
                    skipped[1] += 1
                    continue 
                
            if a in essentials:
                if rows_atypes[i] == 'dma':
                    skipped[2] += 1
                    continue 
            else:
                if rows_atypes[i] == 'tsa':
                    skipped[3] += 1
                    continue
        
            q_smf = rows_queries_smf[i]
            a_smf = rows_arrays_smf[i]
            
            T = rows_temp[i]
            smf_by_gene[a][T].append(a_smf)
            smf_by_gene[q][T].append(q_smf)

            row = {
                "a" : a,
                "b" : q,
                "cs" : rows_cs[i],
                "std" : rows_std[i],
                "gi" : rows_gi[i],
                "a_smf" : rows_arrays_smf[i],
                "b_smf" : rows_queries_smf[i],
                "p_value" : rows_p[i],
                "a_essential" : int(a in essentials),
                "b_essential" : int(q in essentials),
                "temp" : rows_temp[i],
                "atype" : rows_atypes[i],
                "qtype" : rows_qtypes[i]
            }
            rows.append(row)

    illegal_nonessentials = set()
    for node in smf_by_gene:
        smf26 = np.nanmean(smf_by_gene[node][26])
        smf30 = np.nanmean(smf_by_gene[node][30])
        if node not in essentials:
            if not(smf26 >0 or smf30>0):
                illegal_nonessentials.add(node)
    print("Found %s illegal nonessentials" % len(illegal_nonessentials))

    final_df = pd.DataFrame(rows)
    print("Before filtering: ", final_df.shape)

    ix = ~(final_df['a'].isin(illegal_nonessentials) | \
            final_df['b'].isin(illegal_nonessentials))
    final_df = final_df[ix]
    print("After filtering: ", final_df.shape)

    final_df.to_csv('../generated-data/costanzo_gi', index=False)

def get_queries_and_types(df):

    qsid = list(df['Query Strain ID'])

    rows_queries = []
    rows_qtypes = []
    for e in qsid:
        query, strain_id = e.lower().split('_')
        rows_queries.append(nr.get_unified_name(query))
        rows_qtypes.append(re.sub(r'\d+$', '', strain_id))
    
    return rows_queries, rows_qtypes
    
def get_arrays_and_types(df):

    atypes = list(df['Arraytype/Temp'])

    rows_arrays = [nr.get_unified_name(e.lower().split('_')[0]) for e in list(df['Array Strain ID'])]
    rows_atypes = [e.lower().replace('26','').replace('30','') for e in atypes]
    rows_temp = [int(e.lower().replace('tsa', '').replace('dma', '')) for e in atypes]
    
    return rows_arrays, rows_atypes, rows_temp

if __name__ == "__main__":
    main()
    
