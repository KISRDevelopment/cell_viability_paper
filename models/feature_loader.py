import numpy as np 
import json
import numpy.random as rng 


def read_paths(paths):
    Fs = []
    feature_labels = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        Fs.append(d['F'])
        feature_labels.extend(d['feature_labels'].tolist())
    fset = np.hstack(Fs)
    return fset, feature_labels

def load_feature_sets(spec, scramble=False):

    feature_sets = []
    
    for elm in spec:
        paths = elm['paths']

        if not elm['pairwise']:
            fset, feature_labels = read_paths(paths)

            if 'selected_features' in elm:
                selected_features_ix = np.isin(feature_labels, elm['selected_features'])
                fset = fset[:,selected_features_ix]

            if scramble:
                fset = fset[rng.permutation(fset.shape[0]),:]
            
            feature_sets.append(fset)

        else:
            pFs = []
            iu = None
            
            for p in paths:
                pF = np.load(p['path'])

                if len(pF.shape) == 2:
                    pF = np.expand_dims(pF, axis=2)

                if p['normalize']:
                    
                    if iu is None:
                        iu = np.triu_indices(pF.shape[0], 1)

                    # z score the pairwise array
                    for k in range(pF.shape[2]):

                        # get the values in the upper diagonal
                        vals = pF[:,:,k][iu]

                        # compute their mean and std
                        mu = np.mean(vals)
                        std = np.std(vals, ddof=1)
                        #print("Features %s, Mean: %0.4f, Std: %0.4f" % (p['path'], mu, std))
                        
                        # normalize
                        pF[:,:,k] = (pF[:,:,k] - mu) / std 
                

                if 'dummy' in p and p['dummy']:
                    pF = np.ones_like(pF)
                    print("*********** DUMMY PAIRWISE FEATURE ACTIVATED **************")
                pFs.append(pF)

            fset = np.concatenate(pFs, axis=2)

            if scramble:
                fset = fset[rng.permutation(fset.shape[0]),:]
            
            if 'selected_features' in elm:
                fset = fset[:, :, elm['selected_features']]
            
            feature_sets.append(fset)
        
        print("Loaded %s feature set: %s" % (elm['name'], str(fset.shape)))
        
    return feature_sets

if __name__ == "__main__":
    main()
