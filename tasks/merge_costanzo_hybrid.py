import numpy as np 
import pandas as pd 
import sys 
import utils.bin_interacting
import networkx as nx 
def main(gpath, costanzo_path, hybrid_path, output_path):

    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    hybrid_df = pd.read_csv(hybrid_path)

    costanzo_df = pd.read_csv(costanzo_path)
    
    hybrid_pairs = to_dict(hybrid_df)
    print("Hybrid pairs: %d" % len(hybrid_pairs))
    costanzo_pairs = to_dict(costanzo_df)
    print("Costanzo pairs: %d" % len(costanzo_pairs))

    union_pairs = set(hybrid_pairs.keys()).union(costanzo_pairs.keys())

    merged = []
    excluded = 0
    pairs_not_in_costanzo = 0
    pairs_not_in_hybrid = 0
    for pair in union_pairs:
        if pair in costanzo_pairs:
            cbin = costanzo_pairs[pair]
        else:
            cbin = 1
            pairs_not_in_costanzo += 1
        
        if pair in hybrid_pairs:
            hbin = hybrid_pairs[pair]
        else:
            hbin = 1
            pairs_not_in_hybrid += 1

        if (cbin != 1) and (hbin != 1):
            excluded += 1
            continue 
        
        row = {
            "a" : pair[0],
            "b" : pair[1],
            "a_id" : node_ix[pair[0]],
            "b_id" : node_ix[pair[1]], 
            "costanzo_bin" : cbin,
            "hybrid_bin" : hbin
        }
        merged.append(row)
    
    df = pd.DataFrame(merged)

    print("Dataset size: %d" % df.shape[0])
    print("Excluded: %d" % excluded)
    print("Pairs not in costanzo: %d" % pairs_not_in_costanzo)
    print("Pairs not in hybrid: %d" % pairs_not_in_hybrid)

    df.to_csv(output_path, index=False)

def to_dict(df):

    l_a = list(df['a'])
    l_b = list(df['b'])
    l_bin = list(df['bin'])

    return dict([(tuple(sorted((l_a[i], l_b[i]))), l_bin[i]) for i in range(df.shape[0])])

if __name__ == "__main__":
    costanzo_path = sys.argv[1]
    hybrid_path = sys.argv[2]
    output_path = sys.argv[3]
    main(costanzo_path, hybrid_path, output_path)
