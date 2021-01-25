import utils 
import numpy as np 

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

    import db_layer
    import random
    db = db_layer.DbLayer('db.sqlite', 100000)
    
    rows, n_rows = db.get_pairs(1, 0.5, '', 'snf1', 0)
    assert(n_rows > 0)
    sample_rows = random.sample(rows, 50)

    m = LogisticRegressionModel('models/yeast_gi_hybrid_mn.npz')

    for r in sample_rows:

        prob = m.predict(r)

        diff = np.abs(r['prob_gi'] - prob) 

        assert diff < 1e-6

    components = m.get_z_components(rows[10])
    # import json
    # with open('static/interpreation-example.json', 'w') as f:
    #     json.dump(components, f, indent=4)
    print(components)