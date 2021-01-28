import sqlite3
import pandas as pd
import numpy as np 
import os 
import json 
import models.gi_mn
import networkx as nx 
import website.utils 
import utils.extract_biogrid_refs
import utils.extract_fb_refs
DB_PATH = "website/db.sqlite"

def map_common_names_yeast():

    G = nx.read_gpickle('../generated-data/ppc_yeast')
    
    full_names = sorted(G.nodes())

    tag_common = [n.split('  ') for n in full_names]
    tag_common = [(a, '' if a == b else b) for a, b in tag_common]

    return dict(zip(full_names, tag_common))

def map_common_names_pombe():
    
    gene_names = '../data-sources/pombe/PomBase2UniProt.csv'
    gene_names_df = pd.read_csv(gene_names, sep='\t', header=None, names=['locus', 'common'])

    tags = gene_names_df['locus'].str.lower()
    common = gene_names_df['common'].fillna('').str.lower()

    tag_common = list(zip(tags, common))

    G = nx.read_gpickle('../generated-data/ppc_pombe')
    nodes = set(G.nodes())
    assert nodes.intersection(tags) == nodes 

    return dict(zip(tags, tag_common))

def map_common_names_human():

    gene_names = '../data-sources/human/gene_names'
    gene_names_df = pd.read_csv(gene_names, sep='\t')

    tags = gene_names_df['Approved symbol'].fillna('').str.lower()
    common = gene_names_df['HGNC ID'].fillna('').str.lower()

    tag_common = list(zip(tags, common))

    G = nx.read_gpickle('../generated-data/ppc_human')
    nodes = set(G.nodes())

    # print(len(nodes.intersection(tags)))
    # print(set(tags) - nodes)
    # print(nodes - set(tags))
    # assert nodes.intersection(tags) == nodes 

    r = dict(zip(tags, tag_common))
    r['nan'] = ['nan','']
    return r
def map_common_names_dro():

    MAP_FILE = "../data-sources/dro/fbgn_NAseq_Uniprot_fb_2020_01.tsv"

    df = pd.read_csv(MAP_FILE, sep='\t', header=4, na_filter=True)
    df = df[df['organism_abbreviation'] == 'Dmel']

    ix = ~pd.isnull(df['primary_FBgn#'])
    df = df[ix]

    tags = df['primary_FBgn#'].str.lower()
    common = df['gene_symbol'].fillna('').str.lower()

    tag_common = list(zip(tags, common))

    G = nx.read_gpickle('../generated-data/ppc_dro')
    nodes = set(G.nodes())
    assert nodes.intersection(tags) == nodes 

    return dict(zip(tags, tag_common))

SPECIES_GENE_NAMES = {
    1: map_common_names_yeast(),
    2: map_common_names_pombe(),
    3: map_common_names_human(),
    4: map_common_names_dro()
}

# utils.extract_biogrid_refs.main(559292, '../tmp/yeast_refs.json')
# utils.extract_biogrid_refs.main(284812, '../tmp/pombe_refs.json')
# utils.extract_biogrid_refs.main(9606, '../tmp/human_refs.json')
# utils.extract_fb_refs.main('../tmp/dro_refs.json')

def load_refs(path, is_yeast=False):
    with open(path, 'r') as f:
        d = json.load(f)
    
    pairs = [tuple(p) for p in d['pairs']]
    if is_yeast:
        pairs = [ (p[0].split('  ')[0], p[1].split('  ')[0]) for p in pairs]
        
    return dict(zip(pairs, d['pubs']))

SPECIES_PUBS = {
    1: load_refs('../tmp/yeast_refs.json', True),
    2: load_refs('../tmp/pombe_refs.json'),
    3: load_refs('../tmp/human_refs.json'),
    4: load_refs('../tmp/dro_refs.json'), 
}

def create_db(path):

    os.remove(path)

    with open('website_db_schema.sql', 'r') as f:
        schema = f.read()
    conn = connect_db(path)
    c = conn.cursor()
    c.executescript(schema)
    conn.commit()

    conn.close()

def connect_db(path):
    conn = sqlite3.connect(path)
    return conn 

def read_gene_features(cfg_path):
    """ loads the gene-wise features from an MN model config """

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    name_to_spec = { s['name'] : s for s in cfg['spec'] }

    d = np.load(name_to_spec['topology']['path'])
    ix = d['feature_labels'] == 'lid'
    lid = d['F'][:,ix].squeeze()
    smf_feature = np.load(name_to_spec['smf']['path'])['F']
    has_smf = np.sum(smf_feature, axis=1) > 0
    # we add has_smf because some genes have no smf reading. So those get value 0.
    # those with smf, would have values 1, 2, or 3.
    smf = np.argmax(smf_feature, axis=1).astype(int) + has_smf
    sgo = np.load(name_to_spec['sgo']['path'])['F'].astype(bool)

    # now handle the pairwise spl feature
    spl_spec = name_to_spec['spl']
    processor_class = getattr(models.gi_mn, spl_spec['processor'])
    spl_processor = processor_class(spl_spec)

    return lid, smf, sgo, spl_processor

def populate_species(conn, cfg_path, gpath, results_path, species_id):

    lid, smf, sgo, spl = read_gene_features(cfg_path)

    node_ix = read_node_ix(gpath)
    
    node_db_ix = populate_gene_features(conn, lid, smf, sgo, node_ix, species_id)

    populate_interactions(conn, results_path, node_db_ix, node_ix, spl, species_id)

def read_node_ix(gpath):

    G = nx.read_gpickle(gpath)

    nodes = sorted(G.nodes())

    node_ix = dict(zip(nodes, range(len(nodes))))

    return node_ix

def populate_gene_features(conn, lid, smf, sgo, node_ix, species_id):
    
    tag_common = SPECIES_GENE_NAMES[species_id]

    smf = smf.tolist()
    lid = lid.tolist()

    c = conn.cursor()
    rows = []
    for gene, gid in node_ix.items():
        row = (
            species_id,
            tag_common[gene][0],
            tag_common[gene][1],
            lid[gid],
            smf[gid],
            website.utils.pack_sgo(sgo[gid,:]),
        )
        rows.append(row)
    
    c.executemany("INSERT INTO genes(species_id, locus_tag, common_name, lid, smf, sgo_terms) VALUES(?,?,?,?,?,?)", rows)
    conn.commit()
    c.execute("SELECT locus_tag, gene_id FROM genes")
    results = list(c.fetchall())
    n_to_id = {e[0]:e[1] for e in results}

    return n_to_id

def populate_interactions(conn, results_path, node_db_ix, node_ix, spl, species_id):
    df = pd.read_csv(results_path)

    ix = pd.isnull(df['b_id']) | pd.isnull(df['a_id'])
    if np.sum(ix) > 0:
        print("Warning: found %d entries with no a_id or b_id" % np.sum(ix))
        df = df[~ix]

    list_a_ids = [ node_ix[e] for e in df['a_id'] ]
    list_b_ids = [ node_ix[e] for e in df['b_id'] ]

    list_a_names = list(df['a_id'])
    list_b_names = list(df['b_id'])

    if species_id == 1:
        list_a_names = [g.split('  ')[0] for g in list_a_names]
        list_b_names = [g.split('  ')[0] for g in list_b_names]
    
    df['a_name'] = list_a_names
    df['b_name'] = list_b_names

    df['a_id'] = list_a_ids
    df['b_id'] = list_b_ids

    list_prob_gi = list(df['prob_gi'])
    list_obs = list(df['observed'])
    list_interacting = list(df['interacting'])

    # verify publication integrity
    pubs = SPECIES_PUBS[species_id]
    pub_df = df[(df['observed'] == 1) & (df['interacting'] == 1)]
    for i in range(pub_df.shape[0]):
        key = tuple(sorted((pub_df.iloc[i]['a_name'], pub_df.iloc[i]['b_name'])))
        assert  (key in pubs), "Key %s,%s not found in pubs" % key 
    print("Publication integrity OK")

    pairwise_spl = spl.transform(df).squeeze().tolist()
    print("SPL: %d" % len(pairwise_spl))
    
    rows = [(species_id, node_db_ix[a], node_db_ix[b], obs == 1, inter == 1, prob, pspl) for a,b,obs,inter,prob, pspl in 
        zip(list_a_names, list_b_names, list_obs, list_interacting, list_prob_gi, pairwise_spl)]
    c = conn.cursor()

    for i, r in enumerate(rows):
        c.execute("INSERT INTO genetic_interactions(species_id, gene_a_id, gene_b_id, observed, observed_gi, prob_gi, spl) VALUES(?, ?, ?, ?, ?, ?, ?)", r)
        key = tuple(sorted((list_a_names[i], list_b_names[i])))
        gi_id = c.lastrowid

        if list_obs[i] == 1 and list_interacting[i] == 1:
            obs_pubs = pubs[key]
            for p in obs_pubs:
                c.execute("INSERT INTO gi_pubs(gi_id, identifier) VALUES(?, ?)", (gi_id, p))
        
    conn.commit()


create_db(DB_PATH)

conn = connect_db(DB_PATH)

populate_species(conn, 'cfgs/models/yeast_gi_mn.json', '../generated-data/ppc_yeast', '../results/preds/yeast_gi_hybrid_mn', 1)

populate_species(conn, 'cfgs/models/pombe_gi_mn.json', '../generated-data/ppc_pombe', '../results/preds/pombe_gi_mn', 2)

populate_species(conn, 'cfgs/models/human_gi_mn.json', '../generated-data/ppc_human', '../results/preds/human_gi_mn', 3)

populate_species(conn, 'cfgs/models/dro_gi_mn.json', '../generated-data/ppc_dro', '../results/preds/dro_gi_mn', 4)


def interpret(path, ref_class, output_path):

    d = np.load(path)

    weights = d['weights']
    biases = d['biases']

    new_biases = biases - biases[ref_class]
    new_weights = weights - np.expand_dims(weights[:, ref_class], 1)
    
    bias = new_biases[0]
    weights = new_weights[:, 0]

    with open('../generated-data/go_ids_to_names.json', 'r') as f:
        goid_names = json.load(f)

    labels = d['labels']
    labels = [process_label(goid_names, l) for l in labels]
    labels = [l[0].upper() + l[1:] for l in labels]

    np.savez(output_path, bias=bias, weights=weights, labels=labels, orig_weights=d['weights'], orig_biases=d['biases'])

def process_label(goid_names, lbl):

    smf_lookup = {
        'smf_0.00.0' : 'Lethal/Lethal',
        'smf_0.01.0' : 'Lethal/Reduced Growth',
        'smf_0.02.0' : 'Lethal/Normal',
        'smf_1.01.0' : 'Reduced Growth/Reduced Growth',
        'smf_1.02.0' : 'Reduced Growth/Normal',
        'smf_2.02.0' : 'Normal/Normal',
        'smf_00' : 'Lethal/Lethal',
        'smf_01' : 'Lethal/Reduced Growth',
        'smf_02' : 'Lethal/Normal',
        'smf_11' : 'Reduced Growth/Reduced Growth',
        'smf_12' : 'Reduced Growth/Normal',
        'smf_22' : 'Normal/Normal',
    }

    if lbl in goid_names:
        return goid_names[lbl]
    elif lbl == 'lid':
        return 'LID'
    elif lbl == 'pident':
        return 'Percent Identity'
    elif lbl.startswith('sgo_both_'):
        
        goid = lbl.replace('sgo_both_', '')
        go_name = goid_names[goid]
        lbl = '%s (both)' % go_name
    elif lbl.startswith('sgo_either_'):
        goid = lbl.replace('sgo_either_', '').replace('_xor','')
        go_name = goid_names[goid]
        lbl = '%s (either)' % go_name
    elif lbl.startswith('sgo_sum_'):
        goid = lbl.replace('sgo_sum_', '')
        go_name = goid_names[goid]
        lbl = '%s (Sum)' % go_name
    elif lbl == 'sum_lid':
        lbl = 'LID (sum)'
    elif lbl.startswith('smf_'):
        lbl = smf_lookup[lbl]
    elif lbl == 'spl':
        lbl = 'Shortest Path Length'

    return lbl 


"""
interpret('../tmp/models/yeast_gi_hybrid_mn.npz', 1, 'website/models/yeast_gi_hybrid_mn')

interpret('../tmp/models/pombe_gi_mn.npz', 1, 'website/models/pombe_gi_mn')
interpret('../tmp/models/human_gi_mn.npz', 1, 'website/models/human_gi_mn')
interpret('../tmp/models/dro_gi_mn.npz', 1, 'website/models/dro_gi_mn')
"""
