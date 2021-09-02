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

# OLD Biogrid (to replicate wu 2014)
BIOGRID_PATH = "../data-sources/biogrid-old/"
PATHS = [
    ("BIOGRID-SYSTEM-Synthetic_Lethality-3.0.64.mitab", 0)
]

def main(taxid, pub_threshold, output_path):
    taxid_str = "taxid:%d" % taxid
    name_extraction_func = getattr(thismodule, "extract_names_taxid%d" % taxid)

    pairs_to_conditions = defaultdict(list)

    for path, condition in PATHS:
        df = pd.read_csv(os.path.join(BIOGRID_PATH, path+'.txt'), sep='\t')
    
        ix = (df['Taxid Interactor A'] == taxid_str) & (df['Taxid Interactor B'] == taxid_str)
        df = df[ix]

        a, b = name_extraction_func(df)

        for i in range(df.shape[0]):
            key = tuple(sorted((a[i], b[i])))
            pairs_to_conditions[key].append(condition)

    # count how often each condition is experienced      
    rows = []
    conditions_to_pairs = defaultdict(set)
    ignored_overlaps = 0
    for pair, conditions in pairs_to_conditions.items():

        # some pairs can be listed under multiple conditions
        # this is ok as long as there is one condition that has
        # higher frequency of appearance than others.
        # So if a pair occurs twice in bin #2, and twice in bin #3, it is discarded.
        # But if it occurs once in #2, and twice in bin #3, it is accepted.
        condition_cnts = [(c,conditions.count(c)) for c in set(conditions)]
        condition_cnts = sorted(condition_cnts, key=lambda p: p[1], reverse=True)
        if len(condition_cnts) > 1 and condition_cnts[0][1] == condition_cnts[1][1]:
            print("(%s, %s): %s" % (pair[0], pair[1], ', '.join([str(e) for e in conditions])))
            ignored_overlaps += 1
            continue 
    
        most_common_condition = condition_cnts[0]
        
        # only accept if it passes pub threshold
        if most_common_condition[1] >= pub_threshold:
            conditions_to_pairs[most_common_condition[0]].add(pair)

    print("Ignored %d pairs with overlaps" % ignored_overlaps)
    print("Condition counts:")
    for c, v in conditions_to_pairs.items():
        print("%d %d" % (c,len(v)))
    
    for bin, pairs in conditions_to_pairs.items():
        for pair in pairs:
            a, b = pair 
            rows.append({
                "a" : a,
                "b" : b,
                "bin" : bin
            })
    
    df = pd.DataFrame(rows)
    bin = np.array(df['bin'])
    print([np.sum(bin == b) for b in [0,1,2,3]])

    df.to_csv(output_path, index=False)
    

def extract_names_taxid559292(df):

    res = utils.yeast_name_resolver.NameResolver()

    a_names = list(df['Alt IDs Interactor A'])
    b_names = list(df['Alt IDs Interactor B'])

    def extract_locus(e):
        return res.get_unified_name(e.split('|')[-1].replace('entrez gene/locuslink:', '').lower())

    return [extract_locus(e) for e in a_names], [extract_locus(e) for e in b_names]
extract_names_taxid4932 = extract_names_taxid559292

def extract_names_taxid284812(df):

    a_names = list(df['Alt IDs Interactor A'])
    b_names = list(df['Alt IDs Interactor B'])

    def extract_locus(e):
        return e.split('|')[-1].replace('entrez gene/locuslink:', '').lower()

    return [extract_locus(e) for e in a_names], [extract_locus(e) for e in b_names]

extract_names_taxid9606 = extract_names_taxid284812

if __name__ == "__main__":
    taxid = int(sys.argv[1])
    pub_threshold = int(sys.argv[2])
    output_path = sys.argv[3]
    main(taxid, pub_threshold, output_path)
    