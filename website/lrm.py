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

SMF_LABELS = ['Lethal', 'Reduced Growth', 'Normal']
LID_LABEL = 'LID'
N_SMF_BINS = 3
N_SMF_COMBS = np.max(list(SMF_MAP.values())) + 1

class LogisticRegressionModel:

    def __init__(self, path):

        d = np.load(path)

        self._weights = d['weights']
        self._labels = d['labels']
        self._bias = d['bias']

        self.n_sgo = 45
        self._sgo_labels = [l.replace(' (Sum)', '') for l in self._labels[N_SMF_COMBS:(N_SMF_COMBS+self.n_sgo)]]

    def predict(self, row):

        a, avec = self._get_gene_features(row, 'gene_a')
        b, bvec = self._get_gene_features(row, 'gene_b')
        joint_vec = self._vectorize_joint_features(row, a, b)
        
        z = self._bias + np.sum(self._weights * joint_vec['features']) 
        prob = 1 / (1 + np.exp(-z))

        return prob

    def interpret(self, row):
        a, avec = self._get_gene_features(row, 'gene_a')
        b, bvec = self._get_gene_features(row, 'gene_b')
        joint_vec = self._vectorize_joint_features(row, a, b)

        z_components = self._weights * joint_vec['features']
        z_components = np.hstack((self._bias, z_components))
        z_labels = ['Bias'] + joint_vec['labels']

        joint_vec['features'] = joint_vec['features'].tolist()
        
        return {
            "gene_a" : avec,
            "gene_b" : bvec,
            "joint" : joint_vec,
            "z" : {
                "labels" : z_labels,
                "features" : z_components.tolist()
            }
        }

    def _get_gene_features(self, row, whichone):

        lid = row['%s_lid' % whichone]
        smf = int(row['%s_smf' % whichone])
        sgo = utils.unpack_sgo(row['%s_sgo' % whichone], self.n_sgo).astype(int)

        gf = {
            "lid" : lid,
            "smf" : smf, 
            "sgo" : sgo
        }

        return gf, self._vectorize_gene_features(gf)
    
    def _vectorize_gene_features(self, gf):

        smf_bin = np.zeros(N_SMF_BINS)
        if gf['smf'] > 0:
            smf_bin[gf['smf']-1] = 1
        
        v = np.hstack((smf_bin, gf['sgo'], [gf['lid']]))
        labels = SMF_LABELS + self._sgo_labels + [LID_LABEL]
        return {
            "features" : v.tolist(),
            "labels" : labels,
        }
    
    def _vectorize_joint_features(self, row, a, b):

        sum_lid = a['lid'] + b['lid']
        spl = row['spl']
        sum_sgo = a['sgo'] + b['sgo']
        
        smf_bin = np.zeros(N_SMF_COMBS)
        smf_key = (a['smf'], b['smf'])
        if smf_key in SMF_MAP:
            smf = SMF_MAP[smf_key]
            smf_bin[smf] = 1

        v = np.hstack((smf_bin, sum_sgo, spl, sum_lid))
        labels = self._labels 
        return {
            "features" : v,
            "labels" : labels.tolist(),
        }
    
if __name__ == "__main__":

    import db_layer
    import random
    import json
    db = db_layer.DbLayer('db.sqlite', 100000)
    
    rows, n_rows = db.get_pairs(1, 0.5, '', 'snf1', 0)
    assert(n_rows > 0)
    sample_rows = random.sample(rows, 50)

    m = LogisticRegressionModel('models/yeast_gi_hybrid_mn.npz')

    for r in sample_rows:

        prob = m.predict(r)

        diff = np.abs(r['prob_gi'] - prob) 

        assert diff < 1e-6

    interpretation = m.interpret(rows[10])
    # import json
    # with open('static/interpreation-example.json', 'w') as f:
    #     json.dump(components, f, indent=4)
    print(json.dumps(interpretation, indent=4))