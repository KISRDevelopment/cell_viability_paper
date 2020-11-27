import pandas as pd 
import numpy as np 
import networkx as nx 
import feature_preprocessing.redundancy
import sys 
from collections import defaultdict

def main(q_org, db_org, path):

    q_gpath = "../generated-data/ppc_%s" % (q_org)
    db_gpath = "../generated-data/ppc_%s" % (db_org)
    
    # homology db
    q_name_func = getattr(feature_preprocessing.redundancy, "%s_get_name" % q_org)
    db_name_func = getattr(feature_preprocessing.redundancy, "%s_get_name" % db_org)

    # query organism
    q_G = nx.read_gpickle(q_gpath)
    q_nodes = sorted(q_G.nodes())
    q_nodes_ix = dict(zip(q_nodes, range(len(q_nodes))))

    # db organism
    db_G = nx.read_gpickle(db_gpath)
    db_nodes = sorted(db_G.nodes())
    db_nodes_ix = dict(zip(db_nodes, range(len(db_nodes))))

    # homology db
    hom = read_blastp_results(path, q_name_func, db_name_func)

    # feature matrix: id of closest gene, pident, ppos, bitscore, redundancy exists
    F = np.zeros((len(q_nodes), 5))
    
    subject_set = defaultdict(int)
    for query_id, q_node in enumerate(q_nodes):

        if q_node in hom:
            closest_ix = np.argmax([r['bitscore'] for r in hom[q_node]])
            closest = hom[q_node][closest_ix]

            subject = closest['subject']
            
            if subject in db_nodes_ix:
                subject_set[subject] += 1
                F[query_id, 0] = db_nodes_ix[subject]
                F[query_id, 1] = closest['pident']
                F[query_id, 2] = closest['ppos']
                F[query_id, 3] = closest['bitscore']
                F[query_id, 4] = 1
    
    print("# query genes with available homology: %d" % np.sum(F[:,4]))
    print("# subjects: %d" % len(subject_set))
    print("# subjects that are targets to more than one query gene: %d" % len([1 for k, v in subject_set.items() if v > 1]))

    np.savez('../generated-data/features/%s_xhomology_%s' % (q_org, db_org), 
        F = F,
        labels=['subject_id', 'pident', 'ppos', 'bitscore', 'hashom']
    )
    
def read_blastp_results(path, q_name_func, db_name_func):

    results = defaultdict(list)

    with open(path, 'r') as f:

        for line in f:
            query, subject, nident, positive, mismatch, gaps, gapopen, length, pident, ppos, evalue, bitscore = line.strip().split('\t')

            query = q_name_func(query)
            subject = db_name_func(subject)

            results[query].append({
                'subject': subject,
                'pident' : float(pident),
                'ppos' : float(ppos),
                'evalue' : float(evalue),
                'bitscore' : float(bitscore)
            })

    return results

if __name__ == "__main__":
    
    main(*sys.argv[1:])
