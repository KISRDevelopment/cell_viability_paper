import json 
import numpy as np 
import pandas as pd
import igraph as ig

class FeatureMaker:
    
    def __init__(self, ppc_path, sg_path):
        G = ig.read(ppc_path)
        G_node_ix = { l: i for i, l in enumerate(G.vs['label']) }
        
        sorted_nodes = sorted(G_node_ix.keys())
        nix_to_gix = np.array([G_node_ix[n] for n in sorted_nodes])
        
        df = pd.read_feather(sg_path)
        df = df.set_index('id')
        df_node_ix = dict(zip(df['gene'], df.index))
        
        target_node_ix = [G_node_ix[n] for n in df['gene']]
        
        self.G = G
        self.G_node_ix = G_node_ix 
        self.nix_to_gix = nix_to_gix 
        
        self.df = df 
        self.df_node_ix = df_node_ix 
        
        self.target_node_ix = target_node_ix
        
        self.sgo_cols = df.columns[df.columns.str.startswith('sgo-')]

    def make(self, gene_a_id, b_id=None):
        df = self.df 
        sgo_cols = self.sgo_cols
        if b_id is None:
            b_id = np.array(df.index)
        a_id = np.ones(b_id.shape[0], dtype=int)*gene_a_id
        
        
        sum_lid = df.loc[a_id][['topology-lid']].to_numpy() + df.loc[b_id][['topology-lid']].to_numpy()
        sum_sgo = df.loc[a_id][sgo_cols].to_numpy() + df.loc[b_id][sgo_cols].to_numpy()

        smf_a = df.loc[a_id][['bin']].to_numpy()
        smf_b = df.loc[b_id][['bin']].to_numpy()
        
        gi_smf = np.hstack((smf_a, smf_b)).astype(int)
        
        nan_ix = (np.isnan(smf_a) | np.isnan(smf_b))[:,0]
        
        gi_smf = gi_smf[~nan_ix,:]
        gi_smf = np.sort(gi_smf, axis=1)

        rindex_map = np.array([
            [0, 1, 2],
            [1, 3, 4],
            [2, 4, 5]
        ])

        rindex = rindex_map[gi_smf[:,0], gi_smf[:,1]]
        smf_combs = np.zeros((b_id.shape[0], np.max(rindex_map)+1))
        smf_combs[~nan_ix, rindex] = 1   
        
        spl = np.array(self.G.shortest_paths_dijkstra(source=self.nix_to_gix[gene_a_id], 
                                                 target=self.nix_to_gix[b_id])[0])
        spl[np.isinf(spl)] = 1e5
        spl = spl[:,None]
        
        # make sure the order of the features matches the order expected by the model
        F = np.hstack((np.ones_like(spl), spl, sum_lid, smf_combs, sum_sgo))
        
        return F, a_id, b_id 
    
    def get_spl(self, F):
        return F[:, 1]
    
    def get_single_gene_features(self, gid):
        df = self.df 
        smf_cols = df.columns[df.columns.str.startswith('smf-')].tolist()
        lid_cols = ['topology-lid']
        sgo_cols = df.columns[df.columns.str.startswith('sgo-')].tolist()
        
        features =  lid_cols + smf_cols + sgo_cols 

        return features, np.array(df.loc[gid, features])


class TripletFeatureMaker:
    
    def __init__(self, ppc_path, sg_path):
        G = ig.read(ppc_path)
        G_node_ix = { l: i for i, l in enumerate(G.vs['label']) }
        
        sorted_nodes = sorted(G_node_ix.keys())
        nix_to_gix = np.array([G_node_ix[n] for n in sorted_nodes])
        
        df = pd.read_feather(sg_path)
        df = df.set_index('id')
        df_node_ix = dict(zip(df['gene'], df.index))
        
        target_node_ix = [G_node_ix[n] for n in df['gene']]
        
        self.G = G
        self.G_node_ix = G_node_ix 
        self.nix_to_gix = nix_to_gix 
        
        self.df = df 
        self.df_node_ix = df_node_ix 
        
        self.target_node_ix = target_node_ix
        
        self.sgo_cols = df.columns[df.columns.str.startswith('sgo-')]

    def make(self, gene_a_id, gene_b_id, c_id=None):
        df = self.df 
        sgo_cols = self.sgo_cols
        if c_id is None:
            c_id = np.array(df.index)
            
        a_id = np.ones(c_id.shape[0], dtype=int)*gene_a_id
        b_id = np.ones(c_id.shape[0], dtype=int)*gene_b_id
        
        
        sum_lid = df.loc[a_id][['topology-lid']].to_numpy() + \
                  df.loc[b_id][['topology-lid']].to_numpy() + \
                  df.loc[c_id][['topology-lid']].to_numpy()
        
        sum_sgo = df.loc[a_id][sgo_cols].to_numpy() + \
                  df.loc[b_id][sgo_cols].to_numpy() + \
                  df.loc[c_id][sgo_cols].to_numpy()

        smf_a = df.loc[a_id][['bin']].to_numpy()
        smf_b = df.loc[b_id][['bin']].to_numpy()
        smf_c = df.loc[c_id][['bin']].to_numpy()
        
        gi_smf = np.hstack((smf_a, smf_b, smf_c)).astype(int)
        
        nan_ix = (np.isnan(smf_a) | np.isnan(smf_b) | np.isnan(smf_c))[:,0]

        gi_smf = gi_smf[~nan_ix,:]
        gi_smf = np.sort(gi_smf, axis=1)
        
        rindex_map = np.array(
            [[[0, 1, 2], [1, 3, 4], [2, 4, 5]],
             [[1, 3, 4], [3, 6, 7], [4, 7, 8]],
             [[2, 4, 5], [4, 7, 8], [5, 8, 9]]])

        rindex = rindex_map[gi_smf[:,0], gi_smf[:,1], gi_smf[:,2]]
        smf_combs = np.zeros((c_id.shape[0], np.max(rindex_map)+1))
        smf_combs[~nan_ix, rindex] = 1   
        
        spl_ab = np.array(self.G.shortest_paths_dijkstra(source=self.nix_to_gix[gene_a_id], 
                                                         target=self.nix_to_gix[gene_b_id])[0])
        spl_ac = np.array(self.G.shortest_paths_dijkstra(source=self.nix_to_gix[gene_a_id], 
                                                         target=self.nix_to_gix[c_id])[0])
        spl_bc = np.array(self.G.shortest_paths_dijkstra(source=self.nix_to_gix[gene_b_id], 
                                                         target=self.nix_to_gix[c_id])[0])
        
        scl = spl_ab + spl_ac + spl_bc 
        
        scl = scl[:,None]
        
        F = np.hstack((np.ones_like(scl), 
                       sum_sgo,
                       sum_lid, 
                       scl,
                       smf_combs))
        
        return F, a_id, b_id, c_id 
    
class NameMapper:
    
    def __init__(self, path):
        with open(path, 'r') as f:
            name_map = json.load(f)
        
        self.locus_to_id = { l: i for i, l in enumerate(name_map['locus']) }
        self.common_to_id = { c: i for i, c in enumerate(name_map['common']) }
        self.id_to_common = { i: c for i, c in enumerate(name_map['common']) }
        self.id_to_locus = { i: l for i, l in enumerate(name_map['locus']) }
    
    def get_id(self, name):
        if name in self.locus_to_id:
            return self.locus_to_id[name]
        elif name in self.common_to_id:
            return self.common_to_id[name]
        else:
            return None
    
    def get_common(self, id):
        return self.id_to_common.get(id, None)
    
    def get_locus(self, id):
        r = self.id_to_locus.get(id, None)
        if r is not None:
            return r.split('  ')[0]
        return r

class LRModelEnsemble:
    
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.W = d['W']
        self.mu = d['mu']
        self.std = d['std']
        self.features = d['features']
    
    def predict(self, X, return_mean_terms=False):
        
        X = (X - self.mu) / self.std

        # (NxR)
        logit = np.dot(X, self.W)
        
        # (NxR)
        probs = 1/(1+np.exp(-logit))
        
        # (N,)
        mean_probs = np.mean(probs, axis=1)
        
        if return_mean_terms:
            return mean_probs, (X * np.mean(self.W, axis=1))[0]
        
        return mean_probs

class DbLayer:

    def __init__(self):

        self._makers = {
            1: FeatureMaker("data/ppc_yeast.gml", "data/dataset_yeast_allppc.feather"),
            2: FeatureMaker("data/ppc_pombe.gml", "data/dataset_pombe_smf.feather"),
            3: FeatureMaker("data/ppc_human.gml", "data/dataset_human_smf.feather"),
            4: FeatureMaker("data/ppc_dro.gml", "data/dataset_dro_smf.feather")
        }
        
        self._names = {
            1 : NameMapper("data/yeast.json"),
            2 : NameMapper("data/pombe.json"),
            3 : NameMapper("data/human.json"),
            4 : NameMapper("data/dro.json")
        }
        
        self._models = {
            1: LRModelEnsemble("models/gi_yeast.npz"),
            2: LRModelEnsemble("models/gi_pombe.npz"),
            3: LRModelEnsemble("models/gi_human.npz"),
            4: LRModelEnsemble("models/gi_dro.npz"),
        }

        with open('data/go_ids_to_names.json', 'r') as f:
            goid_names = json.load(f)
        
        self.goid_names = goid_names
    
    def get_pairs(self, species_id, threshold, gene_a, gene_b, published_only=False, max_spl = "inf"):
        
        names = self._names[species_id]
        maker = self._makers[species_id]
        model = self._models[species_id]

        gene_a_id = names.get_id(gene_a)
        gene_b_id = names.get_id(gene_b)

        b_id = None
        if gene_a_id is None and gene_b_id is None:
            return [], 0
        elif gene_a_id is not None and gene_b_id is not None:
            b_id = np.array([gene_b_id])
        
        if max_spl == "inf":
            max_spl = 1e5
        max_spl = int(max_spl)

        F, a_id, b_id = maker.make(gene_a_id, b_id)
        preds = model.predict(F)


        spl = maker.get_spl(F)
        ix = (spl <= max_spl) & (preds >= threshold)

        spl = spl[ix]
        a_id = a_id[ix]
        b_id = b_id[ix]
        preds = preds[ix]
        
        rows = [
            {
                "gene_a_id" : a_id[i],
                "gene_b_id" : b_id[i],
                "species_id" : species_id,
                "gene_a_locus_tag" : names.get_locus(gene_a_id),
                "gene_b_locus_tag" : names.get_locus(b_id[i]),
                "gene_a_common_name" : names.get_common(gene_a_id),
                "gene_b_common_name" : names.get_common(b_id[i]),
                "observed" : False,
                "observed_gi" : False,
                "prob_gi" : preds[i],
                "spl" : spl[i]
            }
            for i in range(preds.shape[0])
        ]
        
        return sorted(rows, key=lambda r: r['prob_gi'], reverse=True), len(rows)
    
    def get_gi(self, species_id, gene_a_id, gene_b_id):
        names = self._names[species_id]
        maker = self._makers[species_id]
        model = self._models[species_id]

        F, _, _ = maker.make(gene_a_id, np.array([gene_b_id]))
        preds, mean_logit = model.predict(F, return_mean_terms=True)
        
        # joint features
        joint_features = F[0, :].tolist()
        labels = model.features 
        
        # individual gene features
        single_features, gene_a_features = maker.get_single_gene_features(gene_a_id)
        _, gene_b_features = maker.get_single_gene_features(gene_b_id)
        
        return {
            "gene_a_id" : gene_a_id,
            "gene_b_id" : gene_b_id,
            "species_id" : species_id,
            "prob_gi" : preds[0],
            "joint" : {
                "labels" : self.process_labels(labels),
                "features" : joint_features,
            },
            "z" : {
                "labels" : self.process_labels(labels),
                "features" : mean_logit
            },
            "gene_a" : {
                "labels" : self.process_labels(single_features),
                "features" : gene_a_features
            },
            "gene_b" : {
                "labels" : self.process_labels(single_features),
                "features" : gene_b_features
            },
            "gene_a_locus_tag" : names.get_locus(gene_a_id),
            "gene_b_locus_tag" : names.get_locus(gene_b_id),
            "gene_a_common_name" : names.get_common(gene_a_id),
            "gene_b_common_name" : names.get_common(gene_b_id),
            "pubs" : []
        }
    
    def process_labels(self, labels):
        lookup = {
            'pairwise-spl' : 'Shortest Path Length',
            'topology-lid' : 'LID',
            'smf-LL' : 'Lethal/Lethal',
            'smf-LR' : 'Lethal/Reduced',
            'smf-LN' : 'Lethal/Normal',
            'smf-RR' : 'Reduced/Reduced',
            'smf-RN' : 'Reduced/Normal',
            'smf-NN' : 'Normal/Normal',
            'smf-L' : 'Lethal',
            'smf-R' : 'Reduced',
            'smf-N' : 'Normal',
        }
        processed = []
        for lbl in labels:
            if lbl.startswith('sgo-'):
                v = self.goid_names[lbl.replace('sgo-','')].title()
            else:
                v = lookup.get(lbl, lbl)
            processed.append(v)
        return processed

    def get_common_interactors(self, species_id, genes, threshold, published_only, max_spl="inf"):
        names = self._names[species_id]
        maker = self._makers[species_id]
        model = self._models[species_id]

        if max_spl == "inf":
            max_spl = 1e5
        max_spl = int(max_spl)

        all_preds = []
        all_spls = []
        ix = None

        for gene in genes: 
            gene_id = names.get_id(gene)
            if gene_id is None:
                continue 

            F, _, b_id = maker.make(gene_id, None)
            preds = model.predict(F)
            all_preds.append(preds)
        
            spl = maker.get_spl(F)
            all_spls.append(spl)
            if ix is None:
                ix = (spl <= max_spl) & (preds >= threshold)
            else:
                ix = ix & (spl <= max_spl) & (preds >= threshold)
        
        if ix is None:
            return []
        
        all_preds = np.array(all_preds)
        all_spls = np.array(all_spls)

        all_preds = all_preds[:, ix]
        all_spls = all_spls[:, ix]
        b_id = b_id[ix]


        rows = []
        gene_ids = [names.get_id(g) for g in genes]
        
        for i in range(all_preds.shape[1]):
            common_gene_id = b_id[i]
            
            if common_gene_id in gene_ids:
                continue 
            
            interaction_props = []
            for g, gene in enumerate(genes):
                gene_id = names.get_id(gene)
                interaction_props.append({
                    "gene_locus_tag" : names.get_locus(gene_id),
                    "gene_common_name" : names.get_common(gene_id),
                    "spl" : all_spls[g, i],
                    "gi_prob" : all_preds[g, i]
                })
            
            rows.append({
                "ci_locus_tag" : names.get_locus(common_gene_id),
                "ci_common_name" : names.get_common(common_gene_id),
                "interaction_props" : interaction_props
            })
        
        return rows

if __name__ == "__main__":
    import time 
    import pprint 

    layer = DbLayer()

    # start_time = time.time()

    # for r in range(100):
    #     rows, count = layer.get_pairs(3, 0.9, 'myc', None, 0, max_spl=3)
        
    # end_time = time.time()
    # elapsed = (end_time - start_time)
    # print(" Elapsed: %0.2f sec, average: %0.2f sec" % (elapsed, elapsed/100))
    # print(count)

    #rows, count = layer.get_pairs(3, 0.9, 'myc', None, 0, max_spl=3)
    #print(rows[0])

    r = layer.get_gi(3, 23717, 23603)
    print(r)
    #pprint.pprint(r)

    # rows = layer.get_common_interactors(1, ['snf1', 'snf2', 'spo7'], 0.9, False)
    # pprint.pprint(rows[0])

    #rows, count = layer.get_pairs(3, 0.9, 'myc', None, 0, max_spl=3)
