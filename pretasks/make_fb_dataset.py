import pandas as pd
import numpy as np
from collections import defaultdict

BIOGRID_PATH = '../data-sources/dro/gene_genetic_interactions_fb_2020_01.tsv'

def main(output_path):
    
    df = pd.read_csv(BIOGRID_PATH, sep='\t', header=3)
    ix = ~pd.isnull(df['Starting_gene(s)_FBgn']) & ~pd.isnull(df['Interacting_gene(s)_FBgn'])
    df = df[ix]

    df_sys_a = list(df['Starting_gene(s)_FBgn'].astype(str).apply(get_fb))
    df_sys_b = list(df['Interacting_gene(s)_FBgn'].astype(str).apply(get_fb))
    df_condition = list(df['Interaction_type'])
    
    pair_conds = defaultdict(set)
    for i in range(df.shape[0]):
        a = df_sys_a[i].lower()
        b = df_sys_b[i].lower()
        cond = df_condition[i]
        pair = tuple(sorted((a, b)))
        pair_conds[pair].add(cond)
    
    # only allow pairs associated with one condition
    eligible_pairs = { k: list(v)[0] for k,v in pair_conds.items() if len(v) == 1 }
    
    rows = []
    for p, c in eligible_pairs.items():
        a, b = p 
        rows.append({
            "a" : a,
            "b" : b, 
            "bin" : 0 if c == 'enhanceable' else 2
        })

    df = pd.DataFrame(rows)

    bin = np.array(df['bin'])
    print([np.sum(bin == b) for b in [0,1,2,3]])

    df.to_csv(output_path, index=False)

def get_fb(s):
    parts = s.split('|')
    return parts[0]

if __name__ == "__main__":
    main()
    