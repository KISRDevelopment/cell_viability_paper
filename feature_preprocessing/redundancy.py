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

BLAST_COMMAND = "blastp -query %s -db %s -outfmt '6 qseqid sseqid nident positive mismatch gaps gapopen length pident ppos evalue bitscore' -max_hsps 1 -evalue 0.01 -out %s -seg yes"


def main(organism, gpath):
    
    if organism == "yeast":
        blastp_results_path = "../tmp/blastp_yeast"
        blast_command = BLAST_COMMAND % ("../data-sources/yeast/orf_trans_all.fasta", 
            "../data-sources/yeast/blastdb/sequence", blastp_results_path)
        res = yeast_name_resolver.NameResolver()
        get_name_func = lambda s: yeast_get_name(s, res)
    
    elif organism == "pombe":
        blastp_results_path = "../tmp/blastp_pombe"
        blast_command = BLAST_COMMAND % ("../data-sources/pombe/peptide.fa", 
            "../data-sources/pombe/blastdb/peptide.fa", blastp_results_path)
        get_name_func = pombe_get_name 

    elif organism == "human":
        blastp_results_path = "../tmp/blastp_human"
        blast_command = BLAST_COMMAND % ("../data-sources/human/gencode.v32.pc_translations.fa", 
            "../data-sources/human/blastdb/gencode.v32.pc_translations.fa", blastp_results_path)
        get_name_func = human_get_name 
    
    G = nx.read_gpickle(gpath)
    nodes = list(sorted(G.nodes()))
    
    print("Executing blastp ...")
    #subprocess.run(shlex.split(blast_command))

    blastp_results = read_blastp_results(blastp_results_path, get_name_func)

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
    
def read_blastp_results(path, get_name_func):

    results = defaultdict(list)

    with open(path, 'r') as f:

        for line in f:
            query, subject, nident, positive, mismatch, gaps, gapopen, length, pident, ppos, evalue, bitscore = line.strip().split('\t')

            query = get_name_func(query)
            subject = get_name_func(subject)

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

def yeast_get_name(s, res):
    return res.get_unified_name(s)

def pombe_get_name(s):
    parts = s.split(':')
    return parts[0].lower()

def human_get_name(col):
    parts = col.split('|')
    return parts[6].lower()

if __name__ == "__main__":
    organism = sys.argv[1]
    gpath = sys.argv[2]

    main(organism, gpath)