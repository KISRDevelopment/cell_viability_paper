import pandas as pd 
import numpy as np 
import utils.yeast_name_resolver
import Bio.SeqIO.FastaIO as fio
import networkx as nx
res = utils.yeast_name_resolver.NameResolver()

def main(output_path):

    ess_genes = get_proteins("../data-sources/campos2019/S_cerevisiae_Essential_PROTEINS.txt")
    ness_genes = get_proteins("../data-sources/campos2019/S_cerevisiae_NEssential_PROTEINS.txt")

    print(len(ess_genes))
    print(len(ness_genes))

    print(len(set(ess_genes).intersection(set(ness_genes))))

    G = nx.read_gpickle("../generated-data/ppc_yeast")

    all_genes = set(ess_genes) | set(ness_genes)

    print(len(all_genes.intersection(G.nodes())))

    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))

    rows = [ { "gene" : g, "bin" : 0, "id" : node_ix[g] } for g in ess_genes if g in node_ix ] + \
           [ { "gene" : g, "bin" : 1, "id" : node_ix[g] } for g in ness_genes if g in node_ix ]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(df)
    
def get_proteins(path):


    genes = set()
    with open(path) as handle:
        for title, seq in fio.SimpleFastaParser(handle):
            
            genes.add(title)
    
    genes = res.get_unified_names(genes)
    return genes

if __name__ == "__main__":
    main()
