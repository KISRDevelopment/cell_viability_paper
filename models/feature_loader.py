import numpy as np 
import json
import numpy.random as rng 
import scipy.sparse
import scipy.stats 
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
    fshapes = []

    for elm in spec:
        
        if not elm['pairwise']:
            paths = elm['paths']

            fset, feature_labels = read_paths(paths)

            if 'selected_features' in elm:
                selected_features_ix = np.isin(feature_labels, elm['selected_features'])
                fset = fset[:,selected_features_ix]

            if scramble:
                fset = fset[rng.permutation(fset.shape[0]),:]
            
            feature_sets.append(fset)
            fshapes.append(fset.shape[1:])

        else:
            
            # if elm.get('csr_sparse', False):
            #     F = scipy.sparse.load_npz(elm['path'])
            #     feature_sets.append(F)
            #     fshapes.append((1,))

            #     continue 

            if elm.get('pairwise_sparse', False):
                F = SparsePairwiseMatrix(elm['path'])
                feature_sets.append(F)
                fshapes.append(F.shape())
                continue 
            
            if not elm.get('sparse', False):

                paths = elm['paths']
                
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
                fshapes.append([fset.shape[2],])

            else:
                
                d = np.load(elm['path'])
                indecies = d['indecies']
                data = d['data']

                indexed_data = {}
                for i in range(data.shape[0]):
                    index = tuple(indecies[i, :])
                    indexed_data[index] = data[i,:]

                feature_sets.append(indexed_data)
                print("Loaded %s feature set: %s" % (elm['name'], str(data.shape)))
                fshapes.append([data.shape[1],])

    return feature_sets, fshapes

class SparsePairwiseMatrix(object):

    def __init__(self, path):
        d = np.load(path, allow_pickle=True)


        self.Ps = d['Ps']
        self.node_id_to_comp = d['node_id_to_comp']

        eligible_Ps = [P for P in self.Ps if len(P) > 1]
        n = len(self.node_id_to_comp)
        possible_combs = n * (n-1) / 2 
        avail_combs = sum([len(P)*(len(P)-1)/2 for P in eligible_Ps])

        print("Possible combinations: %d, available: %f" % (possible_combs, avail_combs))

        all_vals = []
        for P in eligible_Ps:
            P = np.array(P)
            iu = np.triu_indices(P.shape[0], 1)
            
            vals = P[iu]
            all_vals.extend(vals)
        
        sum_vals = np.sum(all_vals) + (possible_combs - avail_combs) * 1e5
        self.mu = sum_vals / possible_combs
        self.std = (np.sum(np.square(all_vals - self.mu)) + (possible_combs - avail_combs) * np.square(1e5 - self.mu)) / (possible_combs-1)
        self.std = np.sqrt(self.std)

        #print("Mean: %f, std: %f" % (self.mu, self.std))
    def shape(self):
        return (1,)
    
    def transform(self, df):

        a_id = np.array(df['a_id'])
        b_id = np.array(df['b_id'])

        F = np.zeros((df.shape[0], 1))

        # things that belong to diff components have _infinite_ distance 
        ix_same_comp = self.node_id_to_comp[a_id, 0] == self.node_id_to_comp[b_id, 0]
        F[~ix_same_comp] = 1e5

        # get things that belong to same components
        ix_same_comp = np.where(ix_same_comp)[0]
        for idx in ix_same_comp:
            
            # get the node ids
            a, b = a_id[idx], b_id[idx]
            
            assert self.node_id_to_comp[a,0] == self.node_id_to_comp[b,0]

            # get the corresponding distance matrix
            P = self.Ps[self.node_id_to_comp[a,0]]

            # get their index within the distance matrix
            a_idx = self.node_id_to_comp[a,1]
            b_idx = self.node_id_to_comp[b,1]
            assert a_idx != b_idx, "%s %s" % (a, b)

            # ensure that we haven't processed this already
            assert F[idx] == 0

            # set value
            F[idx] = P[a_idx][b_idx]

            assert not np.isnan(F[idx])

        F = (F-self.mu) / self.std 

        #F = scipy.stats.zscore(F)
        #assert np.sum(np.isnan(new_F)) == 0, "Min: %d, Max: %d" % (np.min(F), np.max(F))

        return F 

if __name__ == "__main__":
    main()
