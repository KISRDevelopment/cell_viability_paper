import os
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import sys
from collections import defaultdict
import numpy.random as rng 
import pickle
import utils.yeast_name_resolver
import json 

thismodule = sys.modules[__name__]

BIOGRID_PATH = "../data-sources/biogrid/"
PATHS = [
    ("BIOGRID-SYSTEM-Synthetic_Lethality-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Synthetic_Growth_Defect-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Negative_Genetic-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Dosage_Growth_Defect-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Dosage_Lethality-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Phenotypic_Enhancement-3.4.156.mitab", 0),
    ("BIOGRID-SYSTEM-Phenotypic_Suppression-3.4.156.mitab", 3),
    ("BIOGRID-SYSTEM-Dosage_Rescue-3.4.156.mitab", 3),
    ("BIOGRID-SYSTEM-Synthetic_Rescue-3.4.156.mitab", 3),
    ("BIOGRID-SYSTEM-Positive_Genetic-3.4.156.mitab", 2)
]


def main(taxid, output_path = ""):
    taxid_str = "taxid:%d" % taxid
    name_extraction_func = getattr(thismodule, "extract_names_taxid%d" % taxid)

    pairs_to_pubs = defaultdict(list)

    for path, condition in PATHS:
        df = pd.read_csv(os.path.join(BIOGRID_PATH, path+'.txt'), sep='\t')
    
        ix = (df['Taxid Interactor A'] == taxid_str) & (df['Taxid Interactor B'] == taxid_str)
        df = df[ix]

        a, b = name_extraction_func(df)

        publications = list(df['Publication Identifiers'])
        for i in range(df.shape[0]):
            key = tuple(sorted((a[i], b[i])))
            pairs_to_pubs[key].append(publications[i])

    keys = list(pairs_to_pubs.keys())
    values = [pairs_to_pubs[k] for k in keys]

    if output_path:

        output = {
            "pairs" : keys,
            "pubs" : values
        }
    
        with open(output_path, 'w') as f:
            json.dump(output, f)

    return pairs_to_pubs
    

def extract_names_taxid559292(df):

    res = utils.yeast_name_resolver.NameResolver()

    a_names = list(df['Alt IDs Interactor A'])
    b_names = list(df['Alt IDs Interactor B'])

    def extract_locus(e):
        return res.get_unified_name(e.split('|')[-1].replace('entrez gene/locuslink:', '').lower())

    return [extract_locus(e) for e in a_names], [extract_locus(e) for e in b_names]

def extract_names_taxid284812(df):

    a_names = list(df['Alt IDs Interactor A'])
    b_names = list(df['Alt IDs Interactor B'])

    def extract_locus(e):
        return e.split('|')[-1].replace('entrez gene/locuslink:', '').lower()

    return [extract_locus(e) for e in a_names], [extract_locus(e) for e in b_names]

extract_names_taxid9606 = extract_names_taxid284812

if __name__ == "__main__":
    taxid = int(sys.argv[1])
    output_path = sys.argv[2]
    main(taxid, output_path)
    