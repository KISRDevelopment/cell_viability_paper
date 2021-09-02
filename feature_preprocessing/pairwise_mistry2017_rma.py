import numpy as np 
import pandas as pd 
import scipy.stats as stats
import os 
import utils.yeast_name_resolver
from collections import defaultdict
import networkx as nx 
import dcor 

res = utils.yeast_name_resolver.NameResolver()

def main(gpath):

    rma_df = pd.read_csv("../data-sources/mistry2017/rma")
    rma_df['raw_gene'] = rma_df['Unnamed: 0']

    print(np.sum(rma_df['raw_gene'].str.endswith('_at')))

    print(list(rma_df['raw_gene']))

    annot_df = pd.read_csv("../data-sources/mistry2017/DIPtoAffy_with_additionalAnnotations.tsv", sep='\t')
    print(annot_df.columns)
    print(annot_df.head())

    pid_unitprot = defaultdict(set)
    for r in annot_df.itertuples():
        input_parts = r[-1].split(';')
        output_parts = r[-3].split(';')

        for input_part in input_parts:
            for output_part in output_parts:
                if output_part != '' and input_part != '':
                    pid_unitprot[input_part].add(res.get_unified_name(output_part))
    
    idmap = {}
    for k, vals in pid_unitprot.items():
        if len(vals) > 1:
            print("%s -> %s" % (k, vals))
        elif len(vals) == 1:
            idmap[k] = list(vals)[0]
    
    print(idmap)
    print(len(idmap))

    ix = rma_df['raw_gene'].isin(idmap)
    rma_df = rma_df[ix].copy()
    rma_df['gene'] = [idmap[g] for g in rma_df['raw_gene']]
    expr_cols = [c for c in rma_df.columns if c.startswith('BT')]
    
    G = nx.read_gpickle(gpath)

    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    data = np.array(rma_df[expr_cols])

    F = np.zeros((len(nodes), len(nodes)))
    for i in range(rma_df.shape[0]):
        node_i = rma_df.iloc[i]['gene']
        if node_i not in node_ix:
            continue
        for j in range(i+1, rma_df.shape[0]):
            node_j = rma_df.iloc[j]['gene']
            if node_j in node_ix:
                node_i_idx = node_ix[node_i]
                node_j_idx = node_ix[node_j]
                F[node_i_idx, node_j_idx] = dcor.distance_correlation(data[i,:], data[j,:])
                F[node_j_idx, node_i_idx] = F[node_i_idx, node_j_idx]
        print("Finished %d" % i)
    
    #print(np.sum(F))
    output_path = "../generated-data/pairwise_features/%s_rma_dcor" % (os.path.basename(gpath))
    
    np.save(output_path, F)
if __name__ == "__main__":
    main("../generated-data/ppc_yeast")