import numpy as np 
import json
import numpy.random as rng 
import models.gi_mn 
import models.gi_nn 

model_classes = {
    "mn" : models.gi_mn,
    "nn" : models.gi_nn
}
class XhomologyFeatureLoader(object):

    def __init__(self, q_org, db_org, db_model_class, db_model_cfg_path, **kwargs):
        self.q_org = q_org
        self.db_org = db_org

        # load target organism's prediction model
        db_model_cfg = load_cfg(db_model_cfg_path, **kwargs)
       
        model_class = model_classes[db_model_class]
        db_model, db_processors = model_class.main(db_model_cfg, 0, 0, '../tmp/dummy', return_model=True)
        self.db_predictor = create_predictor(db_model, db_processors)

        # load homology features
        features_path = "../generated-data/features/%s_xhomology_%s.npz" % (q_org, db_org)
        d = np.load(features_path)
        self.hom = d['F']
        self.labels = d['labels']

        # normalize bitscore
        ix = self.hom[:, 4] > 0
        mean_bitscore = np.mean(self.hom[ix, 3])
        std_bitscore = np.std(self.hom[ix, 3], ddof=1)
        self.hom[ix, 3] = (self.hom[ix, 3] - mean_bitscore) / std_bitscore

    def load(self, df):
        
        # get the homology genes
        hom_aid = self.hom[df['a_id'], 0]
        hom_bid = self.hom[df['b_id'], 0]
        db_df = pd.DataFrame({ "a_id" : hom_aid, "b_id" : hom_bid }).astype(int)

        # get minimum/maximum pident, ppos, bitscore
        min_pident = np.minimum(self.hom[df['a_id'], 1], self.hom[df['b_id'], 1])
        min_ppos = np.minimum(self.hom[df['a_id'], 2], self.hom[df['b_id'], 2])
        min_bitscore = np.minimum(self.hom[df['a_id'], 3], self.hom[df['b_id'], 3])
        max_pident = np.maximum(self.hom[df['a_id'], 1], self.hom[df['b_id'], 1])
        max_ppos = np.maximum(self.hom[df['a_id'], 2], self.hom[df['b_id'], 2])
        max_bitscore = np.maximum(self.hom[df['a_id'], 3], self.hom[df['b_id'], 3])

        # predict
        db_preds = self.db_predictor(db_df)

        # is homology available, are they mapping to same gene, if it is what is the model prediction
        F = np.zeros((df.shape[0], 9))
        F[:, 0] = (self.hom[df['a_id'], 4] > 0) & (self.hom[df['b_id'], 4] > 0)
        F[:, 1] = F[:,0] * ((self.hom[df['a_id'], 0] - self.hom[df['b_id'], 0]) == 0)

        active_ix =  F[:, 0] * (1-F[:,1])

        F[:, 2] = active_ix * db_preds[:, 0]
        F[:, 3] = active_ix * min_pident / 100
        F[:, 4] = active_ix * max_pident / 100
        F[:, 5] = active_ix * min_bitscore
        F[:, 6] = active_ix * max_bitscore
        F[:, 7] = active_ix * min_ppos / 100
        F[:, 8] = active_ix * max_ppos / 100

        return F

def load_cfg(path, **kwargs):

    with open(path, 'r') as f:
        cfg = json.load(f)

    cfg['train_model'] = False
    cfg.update(kwargs)
    
    return cfg 

def create_predictor(model, processors):

    def predict(df, batch_size=10000):

        features = []
        for proc in processors:
            features.append(proc.transform(df))
        if len(processors) > 1:
            batch_F = np.hstack(features)
        else:
            batch_F = features 
        
        return model.predict(batch_F, batch_size)

    return predict

if __name__ == "__main__":

    q_org = "yeast"
    db_org = "pombe"
    db_model_cfg_path = "cfgs/models/pombe_gi_refined_model.json"
    db_trained_model_path = "../results/models/pombe_gi_refined"
    loader = XhomologyFeatureLoader(q_org, db_org, "nn", db_model_cfg_path, trained_model_path=db_trained_model_path,
        targets_path="../generated-data/targets/task_pombe_gi_bin_interacting.npz"
    )

    import numpy.random as rng 
    import pandas as pd
    chosen_combs = rng.choice(5083, (500, 2), replace=False)
    df = pd.DataFrame(data=chosen_combs, columns=['a_id', 'b_id']).astype(int)
    print(df)

    F = loader.load(df)
    np.set_printoptions(precision=3, suppress=True)
    print(F)
    print(np.sum(F, axis=0))