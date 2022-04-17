import pandas as pd 
import json
import numpy as np 

MAP_FILE = "../data-sources/dro/fbgn_NAseq_Uniprot_fb_2020_01.tsv"

def main():
    print("Creating Dmel Entrez -> FBGN map")

    df = pd.read_csv(MAP_FILE, sep='\t', header=4, na_filter=True)

    print(df.columns)

    df = df[df['organism_abbreviation'] == 'Dmel']

    fbgn = list(df['primary_FBgn#'])
    entrez = list(df['EntrezGene_ID'])

    mapping = {}
    for fb_id, en_id in zip(fbgn, entrez):

        if ~np.isnan(en_id):
            key  = int(en_id)
            if key in mapping:
                print("Warning: key %d already mapped to %s, (new map: %s)" % (key, mapping[key], fb_id))
            
            mapping[int(en_id)] = fb_id
    
    with open('../generated-data/dro_gene_map.json', 'w') as f:
        json.dump(mapping, f, indent=4)

if __name__ == "__main__":
    main()
