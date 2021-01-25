import sqlite3
import utils 
import numpy as np 

JOIN_PART = """
    from genetic_interactions g 
    join genes a on g.gene_a_id = a.gene_id 
    join genes b on g.gene_b_id = b.gene_id
"""

FILTER_PART = """where 
    g.species_id = ? and g.prob_gi > ? and (? = '' or (a.locus_tag = ? or a.common_name = ? or b.locus_tag = ? or b.common_name = ?))"""

GI_SELECT_SQL = """select g.gi_id gi_id,
                          g.species_id species_id,
                          a.locus_tag gene_a_locus_tag, 
                          b.locus_tag gene_b_locus_tag,
                          a.common_name gene_a_common_name,
                          b.common_name gene_b_common_name, 
                          g.observed, 
                          g.observed_gi, 
                          g.prob_gi,
                          a.lid gene_a_lid,
                          b.lid gene_b_lid,
                          a.smf gene_a_smf,
                          b.smf gene_b_smf,
                          a.sgo_terms gene_a_sgo,
                          b.sgo_terms gene_b_sgo,
                          g.spl spl 
""" + JOIN_PART

SPECIES_QUERY = GI_SELECT_SQL + FILTER_PART + "limit ? offset ?"
COUNT_QUERY = "select count(g.gi_id) as n_rows" + JOIN_PART + FILTER_PART

SMF_MAP = {
    (1, 1) : 0,
    (1, 2) : 1,
    (2, 1) : 1,
    (1, 3) : 2,
    (3, 1) : 2,
    (2, 2) : 3,
    (2, 3) : 4,
    (3, 2) : 4,
    (3, 3) : 5
}
def get_genes(conn, species_id):

    c = conn.cursor()
    c.execute("select * from genes where species_id = ?", (species_id,))
    rows = c.fetchall()
    # for r in rows:
    #     r['sgo_terms'] = website.utils.unpack_sgo(r['sgo_terms'], 45)
    
    print(rows)

def get_pairs(conn, species_id, threshold, gene, page, entries_per_page):
    gene = gene.lower()

    c = conn.cursor()
    c.execute(SPECIES_QUERY, (species_id, threshold, gene, gene, gene, gene, gene, entries_per_page, page * entries_per_page))
    rows = c.fetchall()

    return rows 

def count_pairs(conn, species_id, threshold, gene):
    c = conn.cursor()
    c.execute(COUNT_QUERY, (species_id, threshold, gene, gene, gene, gene, gene))
    n_rows = c.fetchone()['n_rows']
    return n_rows

def get_gi(conn, gi_id):

    c = conn.cursor()
    c.execute(GI_SELECT_SQL + "where g.gi_id = ?", (gi_id,))
    r = c.fetchone()

    return r 

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

class LogisticRegressionModel:

    def __init__(self, path):

        d = np.load(path)

        self._weights = d['weights']
        self._labels = d['labels']
        self._bias = d['bias']

        
        self.n_sgo = 45
        
    def predict(self, row):
        
        vec = self._arrange_features(row)
        
        z = self._bias + np.sum(self._weights * vec) 
        prob = 1 / (1 + np.exp(-z))

        return prob

    def get_z_components(self, row):
        vec = self._arrange_features(row)
        
        z = self._weights * vec

        z = np.hstack((self._bias, z))

        return {
            "bias" : float(self._bias),
            "components" : z.tolist(),
            "labels" : ['Bias'] + self._labels.tolist()
        }

    def _arrange_features(self, row):

        sum_lid = row['gene_a_lid'] + row['gene_b_lid']

        smf_bin = np.zeros(np.max(list(SMF_MAP.values())) + 1)

        smf_key = (row['gene_a_smf'], row['gene_b_smf'])
        
        if smf_key in SMF_MAP:
            smf = SMF_MAP[smf_key]
            smf_bin[smf] = 1

        spl = row['spl']

        sgo = utils.unpack_sgo(row['gene_a_sgo'], self.n_sgo).astype(int) + \
            utils.unpack_sgo(row['gene_b_sgo'], self.n_sgo).astype(int)

        vec = np.hstack((smf_bin, sgo, spl, sum_lid))
        
        return vec 

if __name__ == "__main__":
    conn = sqlite3.connect('website/db.sqlite')
    conn.row_factory = dict_factory

    rows = get_pairs(conn, 1, 0.9, 0, 100000)

    import numpy.random as rng
    sample_rows = rng.choice(rows, size=50, replace=False)

    m = LogisticRegressionModel('website/models/yeast_gi_hybrid_mn.npz')

    for r in sample_rows:

        prob = m.predict(r)

        diff = np.abs(r['prob_gi'] - prob) 

        assert diff < 1e-6

    components = m.get_z_components(rows[10])
    import json
    with open('website/static/interpreation-example.json', 'w') as f:
        json.dump(components, f, indent=4)

    print(get_gi(conn, 0))
    # assume a MN model with two classes and one feature
    # b1, w1 = [1, 4]
    # b2, w2 = [0, -3]

    # x = 0.5
    # z1 = b1 + w1 * x 
    # z2 = b2 + w2 * x 

    # prob_1 = np.exp(z1) / (np.exp(z1) + np.exp(z2))
    # prob_2 = np.exp(z2) / (np.exp(z1) + np.exp(z2))

    # print("Prob 1: %f, 2: %f" % (prob_1, prob_2))

    # new_b1 = b1 - b2 
    # new_w1 = w1 - w2 
    # new_z = new_b1 + new_w1 * x 
    # prob_1 = 1/(1+np.exp(-new_z))
    # print("Prob 1: %f" % prob_1)