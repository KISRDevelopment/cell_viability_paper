import os 
import sys 
import pandas as pd 
import numpy as np 
import networkx as nx 

def main(gpath, cell_task_path, org_task_path, output_path):
    
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    cell_df = pd.read_csv(cell_task_path)
    org_df = pd.read_csv(org_task_path, keep_default_na=False)

    cell_lethals = set(cell_df[cell_df['bin'] == 0]['gene'])
    cell_viables = set(cell_df[cell_df['bin'] != 0]['gene'])
    org_lethals = set(org_df[org_df['bin'] == 0]['gene'])
    org_viables = set(org_df[org_df['bin'] != 0]['gene'])

    all_viables = cell_viables.union( org_viables )
    viables = all_viables - cell_lethals - org_lethals
    
    cell_lethal_rows = [ {"gene" : g, "bin" : 0, "cs" : 0, "std" : 0 } for g in cell_lethals ]
    org_lethal_rows = [ { "gene" : g, "bin" : 1, "cs" : 0, "std" : 0 } for g in org_lethals ]
    viable_rows = [ { "gene" : g, "bin" : 2, "cs" : 0, "std" : 0 } for g in viables ]


    rows = cell_lethal_rows + org_lethal_rows + viable_rows

    df = pd.DataFrame(rows)
    df['id'] = [node_ix[r['gene']] for r in rows]

    print([np.sum(df['bin'] == b) for b in [0, 1, 2]])

    df.to_csv(output_path, index=False) 

if __name__ == "__main__":
    gpath = sys.argv[1]
    cell_task_path = sys.argv[2]
    org_task_path = sys.argv[3]
    output_path = sys.argv[4]

    main(gpath, cell_task_path, org_task_path, output_path)
