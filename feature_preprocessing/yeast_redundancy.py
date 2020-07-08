import networkx as nx
import numpy as np
import pandas as pd
import json
import sys 
import os
import scipy.stats as stats
from collections import defaultdict 
import subprocess 
import shlex

from utils import yeast_name_resolver

res = yeast_name_resolver.NameResolver()


BLAST_COMMAND = "blastp -query ../data-sources/yeast/orf_trans_all.fasta -db ../data-sources/yeast/blastdb/sequence -outfmt '6 qseqid sseqid nident positive mismatch gaps gapopen length pident ppos evalue bitscore' -max_hsps 1 -evalue 0.01 -out ../generated-data/blastp_yeast -seg yes"
blastp_results_path = '../generated-data/blastp_yeast'


def main():
    gpath = sys.argv[1]

    G = nx.read_gpickle(gpath)
    nodes = list(sorted(G.nodes()))
    
    print("Executing blastp ...")
    subprocess.run(shlex.split(BLAST_COMMAND))

    blastp_results = read_blastp_results(blastp_results_path)

    F = []
    cnt = 0
    for node in nodes:
        f = [0, 0, 0]
        if node in blastp_results:
            closest_ix = np.argmax([r['bitscore'] for r in blastp_results[node]])
            closest = blastp_results[node][closest_ix]

            subject = closest['subject']

            f = [closest['pident'], 
                closest['ppos'], 
                closest['bitscore']]
            cnt += 1
        F.append(f)
    
    print("Nodes in blastp: %d" % cnt)
    # normalize
    F = np.array(F)
    mu = np.mean(F, axis=0)
    std = np.std(F, axis=0)

    print(stats.describe(F))

    F = stats.zscore(F, axis=0)

    output_file = '../generated-data/features/%s_redundancy' % (os.path.basename(gpath))
    np.savez(output_file, F=F, feature_labels=['pident', 'ppos', 'bitscore'], mu=mu, std=std)
    
def read_blastp_results(path):

    results = defaultdict(list)

    with open(path, 'r') as f:

        for line in f:
            query, subject, nident, positive, mismatch, gaps, gapopen, length, pident, ppos, evalue, bitscore = line.strip().split('\t')

            query = res.get_unified_name(query)
            subject = res.get_unified_name(subject)

            if query == subject:
                continue

            results[query].append({
                'subject': subject,
                'pident' : float(pident),
                'ppos' : float(ppos),
                'evalue' : float(evalue),
                'bitscore' : float(bitscore)
            })

    return results

if __name__ == "__main__":
    main()