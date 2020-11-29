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

    def load(self, df):
        
        # get the homology genes
        hom_aid = self.hom[df['a_id'], 0]
        hom_bid = self.hom[df['b_id'], 0]
        db_df = pd.DataFrame({ "a_id" : hom_aid, "b_id" : hom_bid }).astype(int)

        # predict
        db_preds = self.db_predictor(db_df)

        # two features: is homology available, are they mapping to same gene, if it is what is the model prediction
        F = np.zeros((df.shape[0], 3))
        F[:, 0] = (self.hom[df['a_id'], 4] > 0) & (self.hom[df['b_id'], 4] > 0)
        F[:, 1] = F[:,0] * ((self.hom[df['a_id'], 0] - self.hom[df['b_id'], 0]) == 0)
        F[:, 2] = F[:, 0] * (1-F[:,1]) * db_preds[:, 0]

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
    chosen_combs = rng.choice(5083, (100, 2), replace=False)
    df = pd.DataFrame(data=chosen_combs, columns=['a_id', 'b_id']).astype(int)
    print(df)

    F = loader.load(df)
    print(F)