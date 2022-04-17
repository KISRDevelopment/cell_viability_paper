import pandas as pd
import numpy as np
import sys
import networkx as nx
import scipy.stats as stats 

ESSENTIALS_PATH = "../data-sources/pombe/NIHMS57310-supplement-Supplementary_table_1.xls"
SMF_PATH = "../data-sources/pombe/michal_paper_s1.xlsx"

CS_COL = "prototroph_glucose_B%d_Normalised.Colony.Size.Median"
STD_COL = "prototroph_glucose_B%d_Normalised.Colony.Size.Standard.Error"
BIOLOGICAL_REPEAT = 1

# min probability of >= 1 to be considered normal
ALPHA = 0.2
CUTOFF = 1.0

def main(gpath, output_path):
    
    G = nx.read_gpickle(gpath)
    nodes = sorted(G.nodes())
    node_ix = dict(zip(nodes, range(len(nodes))))
    admissible_genes = set(nodes)
    
    edf = pd.read_excel(ESSENTIALS_PATH)
    essentials_df = edf[edf['Analysis dataset'] == 'E']
    essential_genes = set([l.lower() for l in essentials_df['Systemic ID']])
    essential_genes = essential_genes.intersection(admissible_genes)

    smf_df = pd.read_excel(SMF_PATH, sheet_name="Glycerol_screen_data_2")
    smf_df = smf_df[~np.isnan(smf_df[CS_COL % BIOLOGICAL_REPEAT])]
    smf_df['gene'] = [l.lower() for l in smf_df['Gene_ID']]
    ix = smf_df['gene'].isin(admissible_genes)
    smf_df = smf_df[ix]

    viables = set(smf_df['gene'])

    common = viables.intersection(essential_genes)
    print("Common between essentials and viables: %d" % len(common))

    nonessential_genes = viables.difference(common)
    smf_df = smf_df[smf_df['gene'].isin(nonessential_genes)]

    essential_genes = essential_genes.difference(common)

    # get colony sizes
    # those are medians but we treat them as averages
    cs = np.array(smf_df[CS_COL % BIOLOGICAL_REPEAT])
    
    # get standard deviations (SEM * sqrt(N))
    # N=2 for Michal's paper
    std = np.array(smf_df[STD_COL % BIOLOGICAL_REPEAT]) * np.sqrt(2)
    
    genes = list(smf_df['gene'])

    lethal_rows = [{ 'gene' : g, 
        'is_lethal' : 1, 
        'cs' : 0, 
        'std' : 0 } for g in essential_genes]
    nonlethal_rows = [{ 
        'gene' : genes[i], 
        'is_lethal' : 0, 
        'cs' : cs[i], 
        'std' : std[i] } for i in range(len(genes))]
    
    rows = lethal_rows + nonlethal_rows
    df = pd.DataFrame(rows)
    
    df['id'] = [node_ix[n] for n in df['gene']]

    # bin the observations
    cs = np.array(df['cs'])
    std = np.array(df['std'])
    nonlethal_ix = cs > 0
    prob_healthy = (1 - stats.norm.cdf(CUTOFF, cs[nonlethal_ix], std[nonlethal_ix])) >= ALPHA
    bins = np.zeros(df.shape[0])
    bins[nonlethal_ix] = prob_healthy + 1
    df['bin'] = bins

    assert len(set(df[nonlethal_ix]['gene']).intersection(df[~nonlethal_ix]['gene'])) == 0

    print([np.sum(bins == b) for b in np.unique(bins)])

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    gpath = sys.argv[1]

    main(gpath)
    